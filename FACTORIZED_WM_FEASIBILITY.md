# Factorized WM feasibility — hướng dẫn Kaggle

Thí nghiệm offline trên **100 câu GSM8K gốc**, dùng experience và teacher trace
đã lưu trong ZIP hiện tại. Không cần upload một ZIP `2RAW` mới, checkpoint LLM,
hay chạy lại LLM. Không tải Transformers/model weights; không FiLM/controller.

Trong notebook Kaggle, chọn GPU **T4 x2**, bật Internet để tải source và các gói
nhỏ, rồi chạy launcher đã lấy từ đúng branch hoặc commit:

```python
RUN_DIR = '/kaggle/input/datasets/ainzkhail/2source2'
SOURCE_REF = 'codex/factorized-wm-feasibility'  # hoặc commit SHA đã có trên remote
MODE = 'full'  # 'smoke' để kiểm tra pipeline
# RESUME_INPUT = '/kaggle/input/.../previous_result.zip'  # tùy chọn
# RESUME_OUTPUT = '/kaggle/working/factorized_wm_feasibility_full_...'  # tùy chọn

exec(compile(open('/kaggle/working/kaggle_factorized_wm_feasibility.py',
                  encoding='utf-8').read(),
             'kaggle_factorized_wm_feasibility.py', 'exec'))
```

Đặt file launcher trong `/kaggle/working` trước khi chạy cell. Source launcher
phải tương ứng với branch/commit đang dùng. Launcher không commit/push; branch
hoặc SHA phải có trên remote trước khi Kaggle fetch được.

`RUN_DIR` nhận ZIP với bất kỳ tên nào, thư mục đã giải nén, hoặc mount chứa một
run gốc duy nhất. Launcher in các mount root và tên file trước mọi download;
sau khi tải source, `factorized_wm_data.resolve_input` tìm run **theo nội dung**,
đọc member ZIP tại chỗ, không giải nén toàn bộ raw experience.
ZIP chỉ chứa report/CV không thay thế được experience gốc. Không yêu cầu
`checkpoint.pt` trong input. Nếu mount chứa nhiều run, chỉ rõ ZIP/thư mục cần dùng.

Source được lấy bằng git init/fetch/checkout vào thư mục riêng trong
`/kaggle/temp`, hỗ trợ branch lẫn SHA. Launcher chuyển cwd về `/kaggle/working`
trước khi làm việc; không xóa thư mục notebook. D train trên `cuda:1`, V train
trên `cuda:0`. Kaggle yêu cầu hai GPU; CPU chỉ cho local khi đặt rõ
`ALLOW_CPU=True`, kèm `WORKING_DIR` và `TEMP_DIR` nếu cần.

## Cấu hình và giao thức

`configs/factorized_wm_feasibility.json` là full config. Launcher ghi config hiệu
lực ra `<OUTPUT>/launcher_config.json` lúc chạy và gọi đúng CLI:

```text
python run_factorized_wm_feasibility.py --input RUN_DIR --output OUTPUT --config CONFIG_JSON [--resume]
```

Split cố định theo **câu hỏi**, seed 42: train/validation/test = **70/15/15**;
mọi state/edge của một câu thuộc cùng split. Scaling dùng 20/40/60/70 câu trong
train; validation/test giữ nguyên. Các seed fit cuối của D2/V2 là 42/43/44,
không tạo split mới. Checkpoint và ngưỡng operating point chọn trên validation.
Cặp composition chính được định trước: D2/V2, SHT, max hidden dim, medium,
hidden gate bật, action embedding bật, seed 42, toàn bộ train. Full hiện dùng
dim32 mỗi layer, medium width128/layers2 và 70 câu train; không đổi cặp này
theo ranking test hoặc diagnostic no-action.

| Nhóm | Full config |
| --- | --- |
| `num_questions`, `seed`, `seeds` | 100; 42; `[42,43,44]` |
| D / V | D0 ridge, D1 MLP, D2 Transformer / V0 logistic, V1 HGB, V2 Transformer |
| `representations`, `hidden_dims` | `S,H,SH,SHT`; `8,16,32` |
| `capacities` | small 64/1, medium 128/2, large 256/3 (width/layers) |
| `scaling_questions` | `[20,40,60,70]` |
| `projection_updates` | 300 |
| `drafter_updates`, `verifier_updates`, `direct_updates` | 1200 / 1200 / 800 |
| `batch_size`, `eval_every`, `early_patience` | 16 / 100 / 4 |
| `h1_min_updates`, `horizon` | 400 / 3; curriculum H1 → H2 → H3 |
| `device_drafter`, `device_verifier` | `cuda:1` / `cuda:0` |
| `max_proposal_tokens`, `extend_size` | 64 / 8 |
| `bootstrap_samples`, `save_every_stage` | 1000 / true |

S dùng surface gaps/history; H dùng hidden đã chiếu; SH kết hợp cả hai; SHT thêm
token identity codes cố định theo vocabulary train, không phải embedding LLM.
Scalar/structure và context vẫn là metadata chung của các representation.
Hidden dim là số chiều **mỗi layer**, trên 3 layer native 7/14/28.
Learned hidden projector là autoencoder **chỉ dùng drafter hidden ở train**,
được freeze trước khi tạo target dynamics; target và rollout dùng chung hệ tọa
độ đã freeze. Teacher dùng làm supervision, không đưa teacher hiện tại vào
input để giả lập inference.

Cached core chỉ gồm hidden đã chiếu + **32 native gaps** (`H+32`). Bốn chiều
history nằm ngoài core đó và được cập nhật bởi `transition_history`, không
copy như hidden/logits cache bất biến. Confidence delta, token-change và
validity được tính từ transition/representation hiện tại; hidden-change ở
frontier vẫn là D prediction vì phép chiếu mất thông tin không khôi phục được
cosine raw hidden chính xác. Hidden-change của prefix được copy là zero.
Confidence ở vùng mutable/block mới được D dự đoán và clamp `[0,1]`, không
coi là metadata cố định đã biết trước.

Runner tạo ablation theo từng trục rồi loại job trùng, không tích Cartesian
của mọi field: full có **33 job D/V**, smoke có **9**, cộng preprocessing/native
decoder và direct-outcome stage. D0/D1/V0/V1 chạy baseline seed 42; các trục
representation, hidden dim/gate, capacity, final seeds và scaling áp dụng cho
D2/V2. Job D2 có hậu tố `_no_action_embedding` đặt learned action embedding
bằng zero và freeze để chẩn đoán, ghi vào `action_condition_ablation.json`.
Quy tắc R/E về length/commit vẫn hoạt động nên đây không là bằng chứng causal
action-blind; cặp medium SHT có action embedding vẫn là primary.
`experiment_plan.json` là danh sách job chính xác.

Các khóa runner đã công bố là toàn bộ khóa cấp đầu trong bảng. `protocol` là
metadata mô tả dataset, split, D/V, projector, native decoder, kế hoạch/cặp
chính, resume, curriculum, teacher support và giới hạn kết luận; không tự
triển khai các ràng buộc đó. `protocol.native_decoder` ghi nhận readout width128
cố định và budget dùng `projection_updates`; `primary_pair` lấy dim/seed từ
config hiệu lực nên cũng đúng ở smoke. `protocol.resume` mô tả hành vi runner;
launcher kiểm tra `study_manifest.json` cùng config và job folder. Runner/data/models
chịu trách nhiệm thực thi. Launcher thêm `mode` để ghi nhận full/smoke; không
tự giảm grid full theo dung lượng GPU/thời gian. Thiếu resource phải được báo.

`MODE='smoke'` dùng 20 câu, split 14/3/3 (đủ câu validation cho HGB), seed 42,
dim 8, representation SHT, scaling `[14]`, capacity medium width32/layers1.
Projection 10 updates; D/V/direct 20; H1 tối thiểu 5; eval mỗi 10. Các trường
khác giữ nguyên. Smoke kiểm tra pipeline, không là kết quả full feasibility.
`save_every_stage=true` ghi ý định lưu; runner hiện luôn lưu checkpoint theo
job/horizon, không dùng flag này để bật/tắt checkpoint.

## Test, output và resume

Trước thí nghiệm, launcher chạy lần lượt test data, metrics, models, launcher, runner:

```text
tests/test_factorized_wm_data.py
tests/test_factorized_wm_metrics.py
tests/test_factorized_wm_models.py
tests/test_factorized_kaggle_launcher.py
tests/test_factorized_wm_runner.py
```

Launcher test dùng fixture local/mock, không tải source hay chạy LLM. Import
launcher không chạy thí nghiệm. Bộ test trên phải có trong source ref đã chọn.

Output nằm ngay dưới `/kaggle/working`, có timestamp **Asia/Bangkok**, hậu tố
`ICT`; nếu hệ thống thiếu timezone database dùng UTC với hậu tố `UTC`. ZIP là
`OUTPUT.with_suffix('.zip')`. Runner chịu trách nhiệm đóng gói cả khi lỗi;
launcher tạo ZIP từ output còn lại nếu ZIP chưa có, kể cả lỗi preflight/test.
ZIP fallback giữ config, log, report và checkpoint model nhỏ; bỏ input cache,
raw experience, cache model và file lớn hơn 128 MiB. ZIP input không nằm trong
output. Lỗi được in cùng link ZIP trước khi raise. `FileLink` dùng **basename**
tương đối với `/kaggle/working`, tránh link proxy có đường dẫn sai.

Các artifact ở root result; ZIP runner có prefix tên output, ZIP fallback có
thể đặt file ngay ở root ZIP. Chỉ coi hoàn tất khi `summary.json.status` là
`complete`, không chỉ vì có ZIP. Các tên report hiện có:

| Nhóm | File |
| --- | --- |
| Audit/reproducibility | `dataset_audit.json`, `dataset_audit.md`, `feature_schema.json`, `config.json`, `study_manifest.json`, `split_manifest.json`, `source_hashes.json`, `experiment_plan.json`, `summary.json` |
| So sánh/ablation | `drafter_model_comparison.json`, `verifier_model_comparison.json`, `representation_ablation.json`, `hidden_bottleneck_ablation.json`, `capacity_ablation.json`, `action_condition_ablation.json`, `seed_variation.json`, `drafter_scaling.json`, `verifier_scaling.json` |
| D | `drafter_h1_by_action.json`, `drafter_rollout_h1_h2_h3.json`, `drafter_sequence_breakdown.json`, `drafter_token_agreement.json`, `drafter_extend_new_block.json` |
| V | `verifier_token_metrics.json`, `verifier_calibration.json`, `verifier_by_length.json`, `verifier_sequence_metrics.json`, `verifier_response_delta.json` |
| Composition | `composition_oracle_vs_learned.json`, `composition_useful_action_R.json`, `composition_useful_action_E.json`, `useful_action_threshold_sweep_R.csv`, `useful_action_threshold_sweep_E.csv` |
| Diagnostics | `identity_copy_prior.json`, `native_decoder_real_state_floor.json`, `paired_rollout_error_growth.json`, `question_bootstrap.json`, `predictability_knn.json` |
| Raw prediction rows | `identity_copy_predictions.jsonl`, `feasibility_validation_predictions.jsonl`, `feasibility_final_predictions.jsonl` |
| Biểu đồ | `data_scaling.png`, `training_curves.png` |
| Kết luận | `FINAL_FEASIBILITY_REPORT.md`, `final_verdict.json` |

File ablation lọc job theo trục tương ứng; xem trường `job` để biết chính xác
các biến cố định. Các file D chi tiết và V token/calibration/by-length chứa
report tổng hợp; `verifier_sequence_metrics.json` nhóm theo chuỗi action.
D/V predictions và `result.json` của từng job nằm trong `jobs/<job>/`.
Report cuối chỉ xuất sau khi job/stage cuối hoàn tất; ZIP partial không bắt
buộc có đủ các report. `error.txt` là lỗi runner, `launcher_error.txt` là lỗi
launcher. Full report chính dùng cặp seed42; job seed43/44 ở `seed_variation.json`,
không tự gộp thành CI đa seed.

Trong cùng session, đặt `RESUME_OUTPUT` bằng đường dẫn tuyệt đối chính xác của
result folder dưới working. Launcher yêu cầu `config.json`,
`study_manifest.json` và `jobs/` có job folder; manifest phải có fingerprint
SHA256, config khớp root config, source và status `running`/`complete`. Runner
kiểm tra fingerprint thực tế khi nhận `--resume`; launcher không tự xác nhận
nội dung input từ digest đã lưu. Không dùng cả hai biến resume cùng lúc; giữ
nguyên MODE. Có thể gọi CLI trực tiếp:

```text
python /kaggle/temp/<source-repo>/run_factorized_wm_feasibility.py --input /kaggle/input/datasets/ainzkhail/2source2 --output /kaggle/working/<previous-result> --config /kaggle/working/<previous-result>/config.json --resume
```

Resume ở **ranh giới job**, không ở optimizer step: có `jobs/<job>/result.json`
thì bỏ qua job đã hoàn tất. Khi fit đã hoàn tất nhưng chưa có result, dùng lại
`best.pt` cùng `training_complete.json` để đánh giá; D0 ridge dùng best đã lưu
ở cuối fit. Thiếu marker hoàn tất thì fit lại, không coi một best giữa chừng là
job hoàn tất. `last.pt` không phục hồi optimizer/RNG/update hiện hành. Frozen
preprocessor và native decoder đã lưu được dùng lại.
Giữ nguyên config cũ, kể cả metadata, mode và device: runner so fingerprint
của toàn bộ config, question IDs, số state/edge và hash nội dung tensor/label/
edge đã load, **cùng hash source code** của runner, data, models và metrics.
`source_hashes.json` ghi lại các hash này; code thay đổi thì resume bị từ chối
trước khi dùng lại job/checkpoint cũ. Pin `SOURCE_REF` vào SHA gốc để resume,
không dùng branch đã cập nhật. Đổi code hoặc config cần output mới và study
mới; không xóa manifest, sửa fingerprint hay tái dùng checkpoint của study
cũ để vượt kiểm tra. Sửa config mẫu mới cũng làm đổi fingerprint nếu dùng nó
thay config gốc.

Save Version mới cần upload **result ZIP/folder trước** thành dataset, đặt
`RESUME_INPUT`; launcher tìm duy nhất root config/`study_manifest.json`/jobs,
kiểm tra ZIP-slip/symlink trước giải nén và copy đến output mới dưới working.
`RUN_DIR` vẫn dùng GSM8K gốc. Không tự tìm result của session trước. Với run
cũ được gọi trực tiếp bằng config chưa có `mode`, dùng CLI và config gốc để
tránh thêm metadata làm đổi fingerprint.

## Đọc kết luận

Trace có scalar/structure, drafter hidden, token/top-k và teacher đã lưu; các
field thực tế phải được audit từ input. `protocol.teacher_support` là
`native_top32_conditional_only`: thiếu exact full normalizer của D, nên chỉ
đánh giá KL conditional trên native top32; không có KL với other mass hay
KL full vocabulary. Thiếu teacher logits/hidden/trace mới thì báo thiếu và
giới hạn metric; không suy ra hoặc dựng lại bằng LLM.

`NativeTopKDecoder` là readout học **chỉ từ D ở train**, dùng cùng budget
`projection_updates` rồi freeze. Nó là **bottleneck bổ sung** ngoài hidden
projector và dynamics: xem `native_decoder_real_state_floor.json` trên state
test thật trước khi quy lỗi top-k/KL/JS cho rollout. STOP-token identity decoder
là cơ chế riêng; OOV tính sai và coverage được báo. kNN là diagnostic theo
representation hiện tại, không là trần predictability nội tại.

Verdict tự động là **provisional**, không có unqualified `GO`. Nếu thiếu paired
CI95 learned-D-vs-prior ở bất kỳ H1/H2/H3, `final_verdict.json` có
`status="insufficient_evidence"`, `verdict=null`; không biến thiếu dữ liệu thành
NO-GO. Khi đủ cả ba CI, upper bound dưới zero ở ít nhất một horizon cho
`GO—WITH REDESIGN`; nếu không có horizon như vậy thì
`NO-GO WITH CURRENT STATE FORMULATION`. Quy tắc không dùng tolerance oracle
0.5 token. Unqualified GO cần người đọc đánh giá riêng D, V, R/E, oracle gap,
decoder floor và evidence về generalization; không được phát tự động.

Verdict tự động chỉ mang tính **exploratory**, không chứng minh bài toán bất
khả thi. Bộ 100 câu đã được dùng thử nhiều lần nên fixed split hiện tại cũng
là exploratory; không gọi là bằng chứng trên holdout hoàn toàn mới. Kết quả
offline không đo speedup speculative decoding hay chất lượng online chưa chạy.
