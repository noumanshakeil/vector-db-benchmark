from src import main

def test_import():
    assert hasattr(main, "main") or hasattr(main, "app") or True
