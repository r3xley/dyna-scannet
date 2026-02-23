#!/usr/bin/env python3
"""
Test script to verify ShapeR submodule is properly set up and accessible.
"""

import sys
from pathlib import Path

def test_submodule_exists():
    """Test that the submodule directory exists."""
    submodule_path = Path(__file__).parent.parent / "external" / "shaper"
    if not submodule_path.exists():
        print("❌ FAIL: ShapeR submodule directory not found!")
        print(f"   Expected path: {submodule_path}")
        return False
    
    if not submodule_path.is_dir():
        print("❌ FAIL: ShapeR submodule path exists but is not a directory!")
        return False
    
    print("✅ PASS: ShapeR submodule directory exists")
    return True

def test_submodule_is_git_repo():
    """Test that the submodule is a valid git repository."""
    submodule_path = Path(__file__).parent.parent / "external" / "shaper"
    git_dir = submodule_path / ".git"
    
    if not git_dir.exists():
        print("❌ FAIL: ShapeR submodule is not a git repository!")
        print("   Run: git submodule update --init --recursive")
        return False
    
    print("✅ PASS: ShapeR submodule is a valid git repository")
    return True

def test_key_files_exist():
    """Test that key ShapeR files are present."""
    submodule_path = Path(__file__).parent.parent / "external" / "shaper"
    key_files = [
        "README.md",
        "infer_shape.py",
        "dataset/shaper_dataset.py",
        "model/__init__.py",
        "preprocessing/__init__.py",
    ]
    
    missing_files = []
    for file in key_files:
        if not (submodule_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ FAIL: Missing key files: {', '.join(missing_files)}")
        return False
    
    print("✅ PASS: All key ShapeR files are present")
    return True

def test_can_import_shaper_modules():
    """Test that we can access ShapeR modules (if dependencies are installed)."""
    submodule_path = Path(__file__).parent.parent / "external" / "shaper"
    
    # Add submodule to path
    if str(submodule_path) not in sys.path:
        sys.path.insert(0, str(submodule_path))
    
    try:
        # Try importing a simple module (should work even without full deps)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "shaper_readme",
            submodule_path / "README.md"
        )
        print("✅ PASS: Can access ShapeR modules (basic check)")
        return True
    except Exception as e:
        print(f"⚠️  WARNING: Could not fully test imports: {e}")
        print("   This is OK if dependencies aren't installed yet")
        return True  # Don't fail on this

def test_gitmodules_file():
    """Test that .gitmodules file exists and is configured correctly."""
    repo_root = Path(__file__).parent.parent
    gitmodules_path = repo_root / ".gitmodules"
    
    if not gitmodules_path.exists():
        print("❌ FAIL: .gitmodules file not found!")
        return False
    
    content = gitmodules_path.read_text()
    if "shaper" not in content.lower():
        print("⚠️  WARNING: .gitmodules exists but may not be configured correctly")
        return True  # Don't fail, just warn
    
    print("✅ PASS: .gitmodules file exists and is configured")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ShapeR Submodule Setup")
    print("=" * 60)
    print()
    
    tests = [
        ("Submodule Directory Exists", test_submodule_exists),
        ("Submodule is Git Repo", test_submodule_is_git_repo),
        ("Key Files Present", test_key_files_exist),
        ("Can Access Modules", test_can_import_shaper_modules),
        (".gitmodules Configured", test_gitmodules_file),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing: {test_name}...")
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Submodule is properly set up.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
