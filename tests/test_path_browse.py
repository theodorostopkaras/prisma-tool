"""Native Load-tab path chooser helpers (no GUI)."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as ap


def test_load_source_from_pick_file_uses_parent():
    with tempfile.TemporaryDirectory() as tmp:
        grid = os.path.join(tmp, 'grid')
        os.makedirs(grid)
        model = os.path.join(grid, 'Model100_50_20_00_10_15.hdf5')
        with open(model, 'w', encoding='utf-8') as fh:
            fh.write('x')
        assert ap._load_source_from_pick(model) == os.path.abspath(grid)


def test_load_source_from_pick_folder_kept():
    with tempfile.TemporaryDirectory() as tmp:
        grid = os.path.join(tmp, 'grid')
        os.makedirs(grid)
        assert ap._load_source_from_pick(grid) == os.path.abspath(grid)


def test_dialog_start_dir_from_file_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        grid = os.path.join(tmp, 'grid')
        os.makedirs(grid)
        model = os.path.join(grid, 'model.hdf5')
        with open(model, 'w', encoding='utf-8') as fh:
            fh.write('x')
        assert ap._dialog_start_dir(model) == os.path.abspath(grid)
        missing = os.path.join(grid, 'no', 'such', 'dir')
        assert ap._dialog_start_dir(missing) == os.path.abspath(grid)
        assert os.path.isdir(ap._dialog_start_dir(''))


def test_resolve_browse_path_keep_file():
    with tempfile.TemporaryDirectory() as tmp:
        cube = os.path.join(tmp, 'cube.fits')
        with open(cube, 'w', encoding='utf-8') as fh:
            fh.write('x')
        assert ap._resolve_browse_path(cube, keep='file') == os.path.abspath(cube)
        assert ap._resolve_browse_path(cube, keep='folder') == os.path.abspath(tmp)
        assert ap._resolve_browse_path(cube, keep='as_picked') == os.path.abspath(cube)


def test_browse_buttons_on_layout():
    ids = {spec['button'] for spec in ap._PATH_BROWSE}
    inputs = {spec['input'] for spec in ap._PATH_BROWSE}
    assert 'btn-browse-dir' in ids
    assert 'btn-browse-obs-fits' in ids
    assert 'btn-browse-cv-path' in ids
    assert 'btn-browse-fit-output' in ids
    assert 'obs-fits-path' in inputs
    assert 'cv-path' in inputs
    assert 'fit-output-dir' in inputs
    layout = ap.app.layout
    markup = str(layout)
    for btn_id in ids:
        assert btn_id in markup


if __name__ == '__main__':
    test_load_source_from_pick_file_uses_parent()
    test_load_source_from_pick_folder_kept()
    test_dialog_start_dir_from_file_and_missing()
    test_resolve_browse_path_keep_file()
    test_browse_buttons_on_layout()
    print('ok')
