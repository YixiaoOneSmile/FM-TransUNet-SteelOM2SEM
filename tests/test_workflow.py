import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from fmtransunet.dataset import build, sampling_weights
from fmtransunet.registration import landmark_transform, propose, warp
from fmtransunet.refinement import refine
from fmtransunet.train import run


class WorkflowTest(unittest.TestCase):
    def test_registration(self):
        rng = np.random.default_rng(4)
        om = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        sem = om[44:172, 32:160].copy()
        anchors = np.array([[40., 50.], [120., 60.], [60., 140.]])
        result = landmark_transform(anchors, anchors - [32., 44.])
        self.assertLess(max(result['residuals_px']), 1e-6)
        guess = propose(om, sem, 1.)
        target_center = np.array(guess['matrix']) @ [96., 108., 1.]
        self.assertFalse(guess['reflected'])
        np.testing.assert_allclose(target_center[:2], [64., 64.], atol=4.)
        out, mask = warp(om, sem.shape, result['matrix'])
        np.testing.assert_array_equal(out, sem)
        corrected, _, diagnostics = refine(om, sem, np.array(result['matrix']), mask)
        np.testing.assert_array_equal(corrected, sem)
        self.assertEqual(diagnostics['model'], 'identity')

    def test_local_refinement_held_out(self):
        om = np.random.default_rng(44).integers(0, 256, (512, 512), dtype=np.uint8)
        sem = om.copy()
        approximate = np.eye(3)
        approximate[:2, 2] = (4, -3)
        corrected, valid, result = refine(om, sem, approximate, np.full(om.shape, 255, np.uint8))
        self.assertEqual(result['model'], 'translation')
        self.assertLess(result['held_out_after']['rmse'], result['held_out_before']['rmse'])
        error = np.abs(corrected[valid > 0].astype(float) - sem[valid > 0])
        self.assertLess(error.mean(), 1.)

    def test_dataset_and_training_smoke(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as entry:
            root = Path(entry)
            rng = np.random.default_rng(23)
            pairs = []
            mask = np.full((272, 272), 255, dtype=np.uint8)
            Image.fromarray(mask).save(root/'mask.png')
            for i in range(3):
                pixels = rng.integers(0, 256, mask.shape, dtype=np.uint8)
                om = f'om_{i}.png'
                sem = f'sem_{i}.png'
                Image.fromarray(pixels).save(root/om)
                Image.fromarray(255-pixels).save(root/sem)
                pairs.append(dict(pair_id=f'pair_{i}', group=f'group_{i}', material='spheroidized_annealed',
                                  pitch_um=.13, om=om, sem=sem, valid_mask='mask.png', accepted=True,
                                  matrix=np.eye(3).tolist(), crop_xy=[[8, 8]]))
            manifest = root/'pairs.json'
            manifest.write_text(json.dumps(pairs))
            build(manifest, root/'built')
            rows = json.loads((root/'built/patches.json').read_text())
            self.assertAlmostEqual(sum(sampling_weights(rows)), 1.)
            saved_sem = np.array(Image.open(root/'built'/rows[0]['sem']))
            np.testing.assert_array_equal(saved_sem, np.asarray(Image.open(root/'sem_0.png'))[8:264,8:264])
            args = SimpleNamespace(patches=root/'built/patches.json', fold=0, seed=42, device='cpu',
                                   parent=None, variant='fm', paper_config=None, smoke=True, output=root/'parent')
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
                args.parent = args.output/'best.pt'
                args.output = root/'fm'
                run(args)
            self.assertTrue((root/'fm/final.pt').is_file())
            parent = torch.load(args.parent, weights_only=True)['model']
            child = torch.load(root/'fm/final.pt', weights_only=True)['model']
            self.assertTrue(torch.equal(parent['in_conv.net.0.weight'], child['in_conv.net.0.weight']))
            self.assertFalse(torch.equal(parent['out_conv.weight'], child['out_conv.weight']))
            pairs[0]['crop_xy'] = [[-1, 8]]
            manifest.write_text(json.dumps(pairs))
            with self.assertRaises(ValueError):
                build(manifest, root/'invalid')
            self.assertFalse((root/'invalid').exists())


if __name__ == '__main__':
    unittest.main()
