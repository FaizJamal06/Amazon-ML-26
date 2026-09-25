import unittest
from ber.normalize import _extract_house_num, _normalize_base

class TestNormalize(unittest.TestCase):
    def test_extract_house_num_match_position(self):
        # Fix A: check the non-house marker using m.start()
        # Test: "plot 4 sector 4" -> house_num "4"
        house_num, _ = _extract_house_num("plot 4 sector 4", "India")
        self.assertEqual(house_num, "4")

        # Test: "sector 4 pune" -> house_num ""
        house_num, _ = _extract_house_num("sector 4 pune", "India")
        self.assertEqual(house_num, "")

        # Test: "112 mg road plot 12" -> house_num "12"
        house_num, _ = _extract_house_num("112 mg road plot 12", "India")
        self.assertEqual(house_num, "12")

    def test_house_number_markers(self):
        # Fix B: Match house-number markers AFTER punctuation stripping
        # Test: "H.No. 12-3-45" -> addr_norm will be "h no 12-3-45"
        norm = _normalize_base("H.No. 12-3-45")
        self.assertEqual(norm, "h no 12-3-45")
        house_num, _ = _extract_house_num(norm, "India")
        self.assertEqual(house_num, "12-3-45")

        # Test: "N° 12" -> addr_norm will be "n 12"
        norm = _normalize_base("N° 12")
        self.assertEqual(norm, "n 12")
        house_num, _ = _extract_house_num(norm, "India")
        self.assertEqual(house_num, "12")

    def test_normalize_base_rtl_replace(self):
        # Fix C: right-to-left replacement fix for protected-number spans
        # Test: "12/3 MG Road, Shop 45-A" -> exactly "12/3 mg road shop 45-a"
        norm = _normalize_base("12/3 MG Road, Shop 45-A")
        self.assertEqual(norm, "12/3 mg road shop 45-a")

    def test_french_bis_ter(self):
        # Fix 6: Handle French bis/ter suffixes
        # Test: "10 bis rue de la paix"
        norm = _normalize_base("10 bis rue de la paix")
        house_num, addr_nums = _extract_house_num(norm, "France")
        self.assertEqual(house_num, "10bis")
        self.assertIn("10bis", addr_nums)

        # Test: "15 ter avenue"
        norm = _normalize_base("15 ter avenue")
        house_num, addr_nums = _extract_house_num(norm, "France")
        self.assertEqual(house_num, "15ter")
        self.assertIn("15ter", addr_nums)

    def test_longest_legal_form(self):
        from ber.normalize import normalize_name
        # Fix 2 & 3: longest match for legal forms and normalized rule-table keys
        name_fields = normalize_name("XYZ PRIVATE LIMITED", "India")
        self.assertEqual(name_fields["legal_form"], "private limited")
        self.assertEqual(name_fields["name_core"], "xyz")

    def test_unknown_script(self):
        # Fix 4: Classify unknown scripts as non-Latin (Other)
        from ber.normalize import detect_script
        self.assertEqual(detect_script("テスト"), "Other")

if __name__ == '__main__':
    unittest.main()
