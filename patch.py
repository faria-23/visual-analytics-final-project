#!/usr/bin/env python3
"""
Emergency Patch Script for Phase 1 Critical Issues
Automatically applies fixes to resolve backend errors

Usage:
    python emergency_patch_phase1.py

Fixes Applied:
    1. lab_to_rgb import issues
    2. Pydantic validation errors
    3. Country detection debugging
"""

import os
import re
from pathlib import Path

class EmergencyPatcher:
    """Emergency patcher for Phase 1 critical issues"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.fixes_applied = []
        self.errors = []
    
    def apply_all_fixes(self):
        """Apply all emergency fixes"""
        print("🚨 EMERGENCY PATCH FOR PHASE 1 CRITICAL ISSUES")
        print("=" * 60)
        
        try:
            self.fix_1_lab_to_rgb_imports()
            self.fix_2_pydantic_validation()
            self.fix_3_country_detection_debug()
            
            print(f"\n✅ EMERGENCY PATCH COMPLETE")
            print(f"   Fixes applied: {len(self.fixes_applied)}")
            print(f"   Errors encountered: {len(self.errors)}")
            
            if self.errors:
                print(f"\n⚠️ ERRORS DURING PATCHING:")
                for error in self.errors:
                    print(f"   ❌ {error}")
            
            if len(self.fixes_applied) >= 3:
                print(f"\n🎉 READY TO TEST: Restart your server and test again!")
            else:
                print(f"\n⚠️ PARTIAL FIX: Some issues may remain")
            
            return len(self.fixes_applied) >= 3
            
        except Exception as e:
            print(f"❌ EMERGENCY PATCH FAILED: {e}")
            return False
    
    def fix_1_lab_to_rgb_imports(self):
        """Fix lab_to_rgb import issues"""
        try:
            print("\n1️⃣ Fixing lab_to_rgb import issues...")
            
            # Check if lab_to_rgb exists in utils/image_utils.py
            utils_file = self.project_root / "utils" / "image_utils.py"
            
            if not utils_file.exists():
                self.errors.append("utils/image_utils.py not found")
                return
            
            # Read the file
            with open(utils_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if lab_to_rgb function exists
            if "def lab_to_rgb(" not in content:
                print("   🔧 Adding lab_to_rgb function to utils/image_utils.py...")
                
                # Add lab_to_rgb function before the last function
                lab_to_rgb_function = '''
def lab_to_rgb(lab: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """Convert LAB to RGB color space."""
    try:
        L, a, b = lab
        
        # Convert LAB to XYZ
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        def lab_to_xyz_component(t):
            if t > 0.206893:
                return t ** 3
            else:
                return (t - 16/116) / 7.787
        
        x = lab_to_xyz_component(fx) * 0.95047
        y = lab_to_xyz_component(fy) * 1.00000
        z = lab_to_xyz_component(fz) * 1.08883
        
        # Convert XYZ to RGB
        r = x *  3.2404542 + y * -1.5371385 + z * -0.4985314
        g = x * -0.9692660 + y *  1.8760108 + z *  0.0415560
        b = x *  0.0556434 + y * -0.2040259 + z *  1.0572252
        
        # Apply inverse gamma correction
        def inverse_gamma_correct(c):
            if c > 0.0031308:
                return 1.055 * pow(c, 1/2.4) - 0.055
            else:
                return 12.92 * c
        
        r = inverse_gamma_correct(r)
        g = inverse_gamma_correct(g)
        b = inverse_gamma_correct(b)
        
        # Clamp to 0-1 range and convert to 0-255
        r = max(0, min(1, r)) * 255
        g = max(0, min(1, g)) * 255
        b = max(0, min(1, b)) * 255
        
        return (int(r), int(g), int(b))
        
    except Exception as e:
        print(f"Error in LAB to RGB conversion: {e}")
        # Fallback: use L component as grayscale
        gray_val = int(max(0, min(100, lab[0])) * 255 / 100) if len(lab) > 0 else 128
        return (gray_val, gray_val, gray_val)
'''
                
                # Add function before the last line
                content = content.rstrip() + lab_to_rgb_function + "\n"
                
                with open(utils_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("   ✅ Added lab_to_rgb function")
            else:
                print("   ✅ lab_to_rgb function already exists")
            
            # Fix imports in color extraction service
            color_extraction_file = self.project_root / "services" / "color_extraction.py"
            
            if color_extraction_file.exists():
                with open(color_extraction_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if lab_to_rgb is in imports
                if "lab_to_rgb" not in content:
                    print("   🔧 Adding lab_to_rgb to color extraction imports...")
                    
                    # Replace import lines
                    content = re.sub(
                        r'from utils\.image_utils import \((.*?)\)',
                        lambda m: f"from utils.image_utils import ({m.group(1).rstrip()}, lab_to_rgb)",
                        content,
                        flags=re.DOTALL
                    )
                    
                    with open(color_extraction_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("   ✅ Updated color extraction imports")
                else:
                    print("   ✅ lab_to_rgb already in color extraction imports")
            
            # Fix imports in app.py
            app_file = self.project_root / "app.py"
            
            if app_file.exists():
                with open(app_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if "lab_to_rgb" not in content:
                    print("   🔧 Adding lab_to_rgb to app.py imports...")
                    
                    # Replace import lines
                    content = re.sub(
                        r'from utils\.image_utils import (.*?)(?=\n)',
                        lambda m: f"from utils.image_utils import {m.group(1).rstrip()}, lab_to_rgb",
                        content
                    )
                    
                    content = re.sub(
                        r'from \.utils\.image_utils import (.*?)(?=\n)',
                        lambda m: f"from .utils.image_utils import {m.group(1).rstrip()}, lab_to_rgb",
                        content
                    )
                    
                    with open(app_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("   ✅ Updated app.py imports")
                else:
                    print("   ✅ lab_to_rgb already in app.py imports")
            
            self.fixes_applied.append("lab_to_rgb imports fixed")
            
        except Exception as e:
            self.errors.append(f"Fix 1 failed: {e}")
            print(f"   ❌ Error: {e}")
    
    def fix_2_pydantic_validation(self):
        """Fix Pydantic validation error"""
        try:
            print("\n2️⃣ Fixing Pydantic validation error...")
            
            app_file = self.project_root / "app.py"
            
            if not app_file.exists():
                self.errors.append("app.py not found")
                return
            
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for problematic psychology_analysis_level
            if '"psychology_analysis_level": None' in content or "'psychology_analysis_level': None" in content:
                print("   🔧 Removing problematic psychology_analysis_level field...")
                
                # Remove the problematic line
                content = re.sub(
                    r'["\']psychology_analysis_level["\']:\s*None,?\s*\n?',
                    '',
                    content
                )
                
                with open(app_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("   ✅ Removed problematic field")
                self.fixes_applied.append("Pydantic validation fixed")
            else:
                print("   ✅ No problematic psychology_analysis_level found")
                self.fixes_applied.append("Pydantic validation already OK")
            
        except Exception as e:
            self.errors.append(f"Fix 2 failed: {e}")
            print(f"   ❌ Error: {e}")
    
    def fix_3_country_detection_debug(self):
        """Add debug logging to country detection"""
        try:
            print("\n3️⃣ Adding country detection debug logging...")
            
            world_map_file = self.project_root / "services" / "world_map_service.py"
            
            if not world_map_file.exists():
                self.errors.append("services/world_map_service.py not found")
                return
            
            with open(world_map_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if debug logging is already present
            if "🔍 DEBUG: Processing" in content:
                print("   ✅ Debug logging already present")
                self.fixes_applied.append("Debug logging already added")
                return
            
            # Add debug logging to _extract_country_colors_from_clusters
            debug_addition = '''
        print(f"🔍 DEBUG: Processing {len(clusters)} clusters")
        print(f"🔍 DEBUG: url_to_country has {len(url_to_country)} entries")
        print(f"🔍 DEBUG: Sample url_to_country: {dict(list(url_to_country.items())[:3])}")
'''
            
            # Find the function and add debug logging
            pattern = r'(def _extract_country_colors_from_clusters.*?\n.*?try:\n)'
            replacement = r'\\1' + debug_addition
            
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            with open(world_map_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✅ Added debug logging to country detection")
            self.fixes_applied.append("Debug logging added")
            
        except Exception as e:
            self.errors.append(f"Fix 3 failed: {e}")
            print(f"   ❌ Error: {e}")

def main():
    """Run emergency patch"""
    patcher = EmergencyPatcher()
    success = patcher.apply_all_fixes()
    
    if success:
        print("\n🎉 EMERGENCY PATCH SUCCESSFUL!")
        print("📋 NEXT STEPS:")
        print("   1. Restart your server: python app.py")
        print("   2. Run quick test: python quick_test_phase1.py")
        print("   3. Check for remaining errors in server logs")
        
        return True
    else:
        print("\n⚠️ EMERGENCY PATCH INCOMPLETE")
        print("🔧 MANUAL FIXES NEEDED:")
        print("   1. Check the error messages above")
        print("   2. Apply missing fixes manually")
        print("   3. Ensure all files exist and are accessible")
        
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✨ Ready to test Phase 1 again! ✨")
    else:
        print("\n🔧 Manual intervention required")