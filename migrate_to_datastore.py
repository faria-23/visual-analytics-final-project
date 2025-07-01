#!/usr/bin/env python3
"""
Migration Script: Move JSON files from data/ to datastore/
Usage: python migrate_to_datastore.py
"""

import shutil
from pathlib import Path
import json

def migrate_to_datastore():
    """Migrate JSON configuration files from data/ to datastore/"""
    
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    datastore_dir = project_root / "datastore"
    
    print("🔄 MIGRATING TO DATASTORE STRUCTURE")
    print("=" * 50)
    
    # Create datastore directory
    datastore_dir.mkdir(exist_ok=True)
    print(f"✅ Created datastore directory: {datastore_dir}")
    
    # List of JSON files to migrate
    json_files_to_migrate = [
        "country_codes.json",
        "world_countries_masterdata.json", 
        "country_svg_coordinates.json"
    ]
    
    migrated_files = []
    missing_files = []
    
    # Migrate each JSON file
    for json_file in json_files_to_migrate:
        source_path = data_dir / json_file
        dest_path = datastore_dir / json_file
        
        if source_path.exists():
            try:
                # Validate JSON before moving
                with open(source_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # This will throw an exception if invalid JSON
                
                # Move the file
                shutil.move(str(source_path), str(dest_path))
                migrated_files.append(json_file)
                print(f"✅ Migrated: {json_file}")
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON in {json_file}: {e}")
                print(f"   File not migrated: {source_path}")
                
            except Exception as e:
                print(f"❌ Error migrating {json_file}: {e}")
                
        else:
            missing_files.append(json_file)
            print(f"⚠️  File not found: {json_file}")
    
    # Check what remains in data directory
    remaining_files = []
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_file():
                remaining_files.append(item.name)
    
    # Create sample country_codes.json if it doesn't exist
    country_codes_path = datastore_dir / "country_codes.json"
    if not country_codes_path.exists():
        print(f"\n📝 Creating sample country_codes.json...")
        sample_data = {
            "countries": [
                {"code": "US", "name": "United States of America"},
                {"code": "DE", "name": "Germany"},
                {"code": "FR", "name": "France"},
                {"code": "GB", "name": "United Kingdom"},
                {"code": "IT", "name": "Italy"},
                {"code": "ES", "name": "Spain"},
                {"code": "CA", "name": "Canada"},
                {"code": "AU", "name": "Australia"},
                {"code": "JP", "name": "Japan"},
                {"code": "CN", "name": "China"},
                {"code": "NL", "name": "Netherlands"},
                {"code": "CH", "name": "Switzerland"},
                {"code": "AT", "name": "Austria"},
                {"code": "BE", "name": "Belgium"},
                {"code": "DK", "name": "Denmark"},
                {"code": "SE", "name": "Sweden"},
                {"code": "NO", "name": "Norway"},
                {"code": "FI", "name": "Finland"}
            ]
        }
        
        with open(country_codes_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Created sample country_codes.json")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 MIGRATION SUMMARY")
    print("=" * 50)
    
    print(f"✅ Migrated files ({len(migrated_files)}):")
    for file in migrated_files:
        print(f"   📄 {file}")
    
    if missing_files:
        print(f"\n⚠️  Missing files ({len(missing_files)}):")
        for file in missing_files:
            print(f"   📄 {file}")
    
    if remaining_files:
        print(f"\n📁 Files remaining in data/ ({len(remaining_files)}):")
        for file in remaining_files:
            file_path = data_dir / file
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                print(f"   🖼️  {file} (image - correct location)")
            elif file_path.suffix.lower() == '.json':
                print(f"   📄 {file} (JSON - should be in datastore/)")
            else:
                print(f"   📄 {file}")
    
    print(f"\n🎯 NEW DIRECTORY STRUCTURE:")
    print(f"   📁 data/ - Images only ({len([f for f in remaining_files if Path(data_dir / f).suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])} images)")
    print(f"   📁 datastore/ - Configuration files ({len(list(datastore_dir.glob('*.json')))} JSON files)")
    
    print(f"\n✅ Migration completed!")
    print(f"🚀 You can now run: python app.py")
    
    return {
        "migrated": migrated_files,
        "missing": missing_files,
        "remaining": remaining_files
    }

if __name__ == "__main__":
    try:
        result = migrate_to_datastore()
        print(f"\n🎉 Ready to use the new structure!")
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()