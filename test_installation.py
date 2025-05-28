#!/usr/bin/env python3
"""
Test script to verify all packages are installed and working correctly
"""

def test_package(package_name, import_statement=None):
    """Test if a package can be imported"""
    try:
        if import_statement:
            exec(import_statement)
        else:
            __import__(package_name)
        print(f"✅ {package_name}: Successfully imported")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️ {package_name}: Import succeeded but error occurred - {e}")
        return True

def main():
    """Test all required packages"""
    print("Testing package installations...\n")
    
    packages = [
        ("streamlit", None),
        ("langchain", None),
        ("langchain.text_splitter", "from langchain.text_splitter import RecursiveCharacterTextSplitter"),
        ("openai", None),
        ("beautifulsoup4", "import bs4"),
        ("requests", None),
        ("python-dotenv", "import dotenv"),
        ("pinecone", None),
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_statement in packages:
        if test_package(package_name, import_statement):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{total_count} packages working correctly")
    
    if success_count == total_count:
        print("🎉 All packages are ready to use!")
    else:
        print("⚠️ Some packages need attention")
    
    # Test specific functionality
    print(f"\n{'='*50}")
    print("Testing specific functionality...\n")
    
    # Test RecursiveCharacterTextSplitter
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text("This is a test text to split into chunks.")
        print("✅ RecursiveCharacterTextSplitter: Working correctly")
        print(f"   - Created {len(chunks)} chunks from test text")
    except Exception as e:
        print(f"❌ RecursiveCharacterTextSplitter: {e}")
    
    # Test Python version
    import sys
    print(f"✅ Python version: {sys.version}")

if __name__ == "__main__":
    main() 