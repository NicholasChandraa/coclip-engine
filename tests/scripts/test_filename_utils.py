
import unittest
from app.utils.filename import sanitize_filename

class TestFilenameUtils(unittest.TestCase):
    def test_sanitize_basic(self):
        self.assertEqual(sanitize_filename("Hello World"), "Hello_World")
        self.assertEqual(sanitize_filename("Clip Title"), "Clip_Title")

    def test_sanitize_special_chars(self):
        self.assertEqual(sanitize_filename("Clip #1: The Best!"), "Clip_1_The_Best")
        self.assertEqual(sanitize_filename("Win/Loss"), "Win_Loss")
        self.assertEqual(sanitize_filename("Dr. Strange"), "Dr._Strange")
        
    def test_sanitize_unicode(self):
        # Depending on regex, this might strip unicode or keep it if \w matches unicode
        # My regex was [^a-zA-Z0-9\-\.] so it strips unicode
        self.assertEqual(sanitize_filename("Café"), "Caf")
        
    def test_sanitize_empty(self):
        self.assertEqual(sanitize_filename("", fallback="fallback"), "fallback")
        self.assertEqual(sanitize_filename("!!!", fallback="fallback"), "fallback")
        
    def test_sanitize_truncation(self):
        long_name = "a" * 100
        sanitized = sanitize_filename(long_name)
        self.assertEqual(len(sanitized), 50)
        
    def test_sanitize_strip(self):
        self.assertEqual(sanitize_filename("  test  "), "test")
        self.assertEqual(sanitize_filename("__test__"), "test")

if __name__ == '__main__':
    unittest.main()
