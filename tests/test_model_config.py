import json

import model_config as mc

NEW_STYLE = '''{
  "physical_params": {
    // Physical parameters of the PDR
    "cosmic_ray_rate": 1e-13,         // Primary cosmic ray ionization rate (in s^-1): COSRAY
    "cr_attenuation_model": "L",      // mode 2: Padovani+2024 model "L" or "H"
    "file": "a//b.fuv",               // slashes inside strings are kept
    "metallicity": 1.0                // Metallicity scaling factor: ZMETAL
  }
}'''


def test_jsonc_config_parses_and_keeps_comments(tmp_path):
    path = tmp_path / 'pdr_config_x.json'
    path.write_text(NEW_STYLE)
    pp = mc.load_config(str(path))['physical_params']
    assert pp['cosmic_ray_rate'] == 1e-13
    assert pp['cr_attenuation_model'] == 'L'
    assert pp['file'] == 'a//b.fuv'
    assert pp['metallicity_comment'] == 'Metallicity scaling factor: ZMETAL'
    assert mc.extract_comment_labels({'physical_params': pp})[
        'physical_params.cosmic_ray_rate'] == 'Primary cosmic ray ionization rate (in s^-1)'


def test_plain_json_config_unchanged(tmp_path):
    data = {'physical_params': {'cr_att_coeff': 0.4, 'cr_att_coeff_comment': 'x: y'}}
    path = tmp_path / 'pdr_config_y.json'
    path.write_text(json.dumps(data))
    assert mc.load_config(str(path)) == data
