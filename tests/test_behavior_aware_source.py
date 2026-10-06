import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from behavior_aware_source import REQUIRED, resolve_phase0_input, materialize_phase0


def source_archive(path,unsafe=None):
    cfg=dict(num_questions=100,latent_dims=[64,128],token_embedding='pretrained')
    split=dict(train=list(range(70)),val=list(range(70,85)),test=list(range(85,100)))
    values={'config.json':cfg,'summary.json':dict(schema='latent_world_model_phase0_v1'),
            'split_manifest.json':split}
    with zipfile.ZipFile(path,'w') as z:
        for name in REQUIRED:z.writestr('arbitrary/'+name,json.dumps(values.get(name,{})))
        if unsafe:z.writestr(unsafe,'bad')


class SourceTests(unittest.TestCase):
    def test_arbitrary_zip_filename_and_already_extracted_folder(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);archive=root/'uploaded.dat';source_archive(archive)
            self.assertEqual(resolve_phase0_input(root),archive.resolve())
            extracted=materialize_phase0(archive,root/'cache')
            self.assertEqual(resolve_phase0_input(extracted),extracted.resolve())
            self.assertTrue(all((extracted/n).exists() for n in REQUIRED))

    def test_no_source_or_two_sources_fail_clearly(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):resolve_phase0_input(root)
            source_archive(root/'one.zip');source_archive(root/'two.zip')
            with self.assertRaises(ValueError):resolve_phase0_input(root)

    def test_unsafe_archive_paths_rejected_before_materialization(self):
        with tempfile.TemporaryDirectory() as d:
            for i,name in enumerate(('../outside.txt','C:/absolute.txt','/absolute.txt','folder/../../escape')):
                p=Path(d)/f'{i}.zip';source_archive(p,name)
                with self.subTest(name=name),self.assertRaises(ValueError):materialize_phase0(p,Path(d)/f'cache{i}')


if __name__=='__main__':unittest.main()
