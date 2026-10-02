from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from persistent_world_model_v1 import (GatedFiLMAdapter, PersistentWorldModelV1,
    expected_yield, grouped_distribution_kl, hazard_mode, hazard_nll)


def _batch(batch=2, length=5, hidden_dim=16, token_dim=12, top_k=4):
    labels = torch.tensor([2, 5][:batch])
    teacher = torch.randn(batch, length, 35)
    valid = torch.zeros(batch, length, dtype=torch.bool)
    valid[0, :length] = True
    if batch > 1:
        valid[1, :length] = True
    return dict(
        hidden=torch.randn(batch, length, 3, hidden_dim),
        token_vectors=torch.randn(batch, length, 2, token_dim),
        topk_vectors=torch.randn(batch, length, token_dim),
        gaps=torch.randn(batch, length, top_k),
        scalars=torch.randn(batch, length, 16),
        history=torch.randn(batch, length, 4),
        prefix_vectors=torch.randn(batch, 7, token_dim),
        prefix_lengths=torch.tensor([7] * batch),
        context=torch.randn(batch, 8),
        lengths=torch.tensor([length] * batch),
        labels=labels,
        teacher_margin=torch.randn(batch, length),
        teacher_valid=valid,
        teacher_features=teacher,
        teacher_aux_features=torch.randn(batch, length, 6),
        teacher_actual=torch.tensor([True] * batch),
    )


def test_hazard_likelihood_uses_only_prefix_through_first_reject():
    logits = torch.tensor([[1.0, 0.5, -0.2, 8.0, -8.0]])
    y = hazard_nll(logits, torch.tensor([5]), torch.tensor([2]))
    changed_suffix = logits.clone()
    changed_suffix[0, 3:] = torch.tensor([-80.0, 80.0])
    assert torch.allclose(y, hazard_nll(changed_suffix, torch.tensor([5]), torch.tensor([2])))
    assert expected_yield(logits, torch.tensor([5])).shape == (1,)
    assert int(hazard_mode(logits)[0]) <= 5


def test_verifier_encoder_masks_suffix_after_first_rejection():
    torch.manual_seed(1)
    model = PersistentWorldModelV1(hidden_dim=16, token_dim=12, top_k=4,
                                   num_hidden_layers=3, dropout=0.0).eval()
    batch = _batch(batch=1)
    # Y=2: verifier features may be consumed only at positions 0,1,2.
    batch["labels"] = torch.tensor([2])
    first = model.encode_v(batch)
    changed = {key: value.clone() if isinstance(value, torch.Tensor) else value
               for key, value in batch.items()}
    changed["teacher_features"][:, 3:, :] = torch.randn_like(changed["teacher_features"][:, 3:, :]) * 100
    changed["teacher_aux_features"][:, 3:, :] = torch.randn_like(changed["teacher_aux_features"][:, 3:, :]) * 100
    second = model.encode_v(changed)
    assert torch.allclose(first, second, atol=1e-6)


def test_verifier_encoder_handles_batch_rows_without_teacher_observations():
    torch.manual_seed(11)
    model = PersistentWorldModelV1(hidden_dim=16, token_dim=12, top_k=4,
                                   num_hidden_layers=3, dropout=0.0).eval()
    batch = _batch(batch=2)
    batch["teacher_valid"][1] = False
    batch["labels"][1] = -1
    encoded = model.encode_v(batch)
    assert torch.isfinite(encoded).all()
    assert torch.equal(encoded[1], torch.zeros_like(encoded[1]))


def test_drafter_encoder_has_no_current_verifier_input():
    torch.manual_seed(2)
    model = PersistentWorldModelV1(hidden_dim=16, token_dim=12, top_k=4,
                                   num_hidden_layers=3, dropout=0.0).eval()
    batch = _batch(batch=1)
    first = model.encode_d(batch)
    changed = dict(batch)
    changed["teacher_features"] = torch.randn_like(batch["teacher_features"]) * 100
    changed["teacher_margin"] = torch.randn_like(batch["teacher_margin"]) * 100
    second = model.encode_d(changed)
    assert torch.equal(first, second)


def test_transition_and_correction_keep_fixed_128d_state():
    model = PersistentWorldModelV1(hidden_dim=16, token_dim=12, top_k=4,
                                   num_hidden_layers=3, dropout=0.0)
    z, d = torch.randn(3, 128), torch.randn(3, 64)
    for action in ("R", "E"):
        result = model.transition(z, action, d)
        assert result.shape == (3, 128)
        assert torch.isfinite(result).all()


def test_film_is_exact_identity_at_initialization_and_trainable():
    torch.manual_seed(3)
    adapter = GatedFiLMAdapter(16)
    hidden, z = torch.randn(2, 5, 16), torch.randn(2, 128)
    output = adapter(hidden, z)
    assert torch.equal(output, hidden)
    (output.float().square().sum()).backward()
    assert adapter.film.weight.grad is not None
    assert adapter.film.weight.grad.abs().sum() > 0


def test_verifier_distillation_uses_topk_plus_other_bucket():
    student = torch.randn(6, 20, requires_grad=True)
    ids = torch.topk(torch.randn(6, 20), 4, dim=-1).indices
    teacher_logits = torch.randn(6, 4)
    logsumexp = torch.logsumexp(torch.randn(6, 20), dim=-1)
    loss = grouped_distribution_kl(student, teacher_logits, logsumexp, ids,
                                   preserve_logits=torch.randn_like(student))
    assert loss.shape == (6,)
    assert torch.isfinite(loss).all()
    loss.mean().backward()
    assert student.grad is not None


if __name__ == "__main__":
    for test in (test_hazard_likelihood_uses_only_prefix_through_first_reject,
                 test_verifier_encoder_masks_suffix_after_first_rejection,
                 test_verifier_encoder_handles_batch_rows_without_teacher_observations,
                 test_drafter_encoder_has_no_current_verifier_input,
                 test_transition_and_correction_keep_fixed_128d_state,
                 test_film_is_exact_identity_at_initialization_and_trainable,
                 test_verifier_distillation_uses_topk_plus_other_bucket):
        test()
    print("persistent world model V1 tests: OK")
