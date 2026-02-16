"""
Unit tests for functions/utils/text.py

Coverage:
- trim_lr
- normalize_ws
- safe_truncate
- to_context_str
"""

import math

import pytest

from functions.utils.text import normalize_ws, safe_truncate, to_context_str, trim_lr


class TestTrimLR:
    """Tests for trim_lr function."""

    def test_trim_leading_whitespace(self):
        assert trim_lr("  hello") == "hello"
        assert trim_lr("\thello") == "hello"
        assert trim_lr("\nhello") == "hello"

    def test_trim_trailing_whitespace(self):
        assert trim_lr("hello  ") == "hello"
        assert trim_lr("hello\t") == "hello"
        assert trim_lr("hello\n") == "hello"

    def test_trim_both_sides(self):
        assert trim_lr("  hello  ") == "hello"
        assert trim_lr("\t\nhello\n\t") == "hello"

    def test_preserve_internal_whitespace(self):
        assert trim_lr("  hello  world  ") == "hello  world"
        assert trim_lr("  hello\n\nworld  ") == "hello\n\nworld"
        assert trim_lr("  hello\t\tworld  ") == "hello\t\tworld"

    def test_empty_string(self):
        assert trim_lr("") == ""

    def test_whitespace_only(self):
        assert trim_lr("   ") == ""
        assert trim_lr("\t\n  ") == ""

    def test_no_whitespace(self):
        assert trim_lr("hello") == "hello"

    def test_unicode_whitespace(self):
        # Regular space + NBSP (U+00A0)
        assert trim_lr("  hello\u00A0") == "hello"


class TestNormalizeWS:
    """Tests for normalize_ws function."""

    def test_collapse_multiple_spaces(self):
        assert normalize_ws("hello    world") == "hello world"
        assert normalize_ws("a  b  c") == "a b c"

    def test_collapse_mixed_whitespace(self):
        assert normalize_ws("hello\n\nworld") == "hello world"
        assert normalize_ws("hello\t\tworld") == "hello world"
        assert normalize_ws("hello \n\t world") == "hello world"

    def test_trim_and_collapse(self):
        assert normalize_ws("  hello    world  ") == "hello world"
        assert normalize_ws("\n\nhello\n\nworld\n\n") == "hello world"

    def test_empty_string(self):
        assert normalize_ws("") == ""

    def test_whitespace_only(self):
        assert normalize_ws("   ") == ""
        assert normalize_ws("\n\t  ") == ""

    def test_no_whitespace(self):
        assert normalize_ws("hello") == "hello"

    def test_single_word_with_whitespace(self):
        assert normalize_ws("  hello  ") == "hello"

    def test_complex_whitespace_pattern(self):
        text = "  hello\n  world\t\t\nfoo   bar  "
        assert normalize_ws(text) == "hello world foo bar"

    def test_preserves_order(self):
        assert normalize_ws("a\nb\nc") == "a b c"


class TestSafeTruncate:
    """Tests for safe_truncate function."""

    def test_no_truncation_needed(self):
        result, applied = safe_truncate("hello", 10)
        assert result == "hello"
        assert applied is False

    def test_exact_length(self):
        result, applied = safe_truncate("hello", 5)
        assert result == "hello"
        assert applied is False

    def test_truncation_applied(self):
        result, applied = safe_truncate("hello world", 5)
        assert result == "hello…"
        assert applied is True

    def test_truncation_strips_trailing_whitespace(self):
        result, applied = safe_truncate("hello     world", 8)
        assert result == "hello…"  # Strips trailing spaces before ellipsis
        assert applied is True

    def test_max_chars_zero(self):
        result, applied = safe_truncate("hello", 0)
        assert result == "hello"
        assert applied is False

    def test_max_chars_negative(self):
        result, applied = safe_truncate("hello", -1)
        assert result == "hello"
        assert applied is False

    def test_custom_ellipsis(self):
        result, applied = safe_truncate("hello world", 5, ellipsis="...")
        assert result == "hello..."
        assert applied is True

    def test_empty_string(self):
        result, applied = safe_truncate("", 5)
        assert result == ""
        assert applied is False

    def test_single_char(self):
        result, applied = safe_truncate("h", 1)
        assert result == "h"
        assert applied is False

    def test_truncate_to_one_char(self):
        result, applied = safe_truncate("hello", 1)
        assert result == "h…"
        assert applied is True

    def test_unicode_ellipsis_default(self):
        result, applied = safe_truncate("hello world", 5)
        assert "…" in result  # Unicode ellipsis
        assert applied is True

    def test_long_text_truncation(self):
        long_text = "a" * 1000
        result, applied = safe_truncate(long_text, 50)
        assert len(result) == 51  # 50 chars + ellipsis
        assert result.endswith("…")
        assert applied is True


class TestToContextStr:
    """Tests for to_context_str function."""

    def test_none_value(self):
        assert to_context_str(None) == ""

    def test_float_nan(self):
        assert to_context_str(float("nan")) == ""
        assert to_context_str(math.nan) == ""

    def test_regular_string(self):
        assert to_context_str("hello") == "hello"
        assert to_context_str("") == ""

    def test_integer(self):
        assert to_context_str(42) == "42"
        assert to_context_str(0) == "0"
        assert to_context_str(-1) == "-1"

    def test_float(self):
        assert to_context_str(3.14) == "3.14"
        assert to_context_str(0.0) == "0.0"

    def test_boolean(self):
        assert to_context_str(True) == "True"
        assert to_context_str(False) == "False"

    def test_list(self):
        assert to_context_str([1, 2, 3]) == "[1, 2, 3]"

    def test_dict(self):
        result = to_context_str({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_complex_object(self):
        class CustomObj:
            def __str__(self):
                return "custom_representation"

        assert to_context_str(CustomObj()) == "custom_representation"

    def test_zero_vs_none(self):
        # Ensure 0 is not treated as None
        assert to_context_str(0) == "0"
        assert to_context_str(None) == ""

    def test_empty_string_vs_none(self):
        assert to_context_str("") == ""
        assert to_context_str(None) == ""

    def test_float_infinity(self):
        assert to_context_str(float("inf")) == "inf"
        assert to_context_str(float("-inf")) == "-inf"


class TestIntegration:
    """Integration tests combining multiple utilities."""

    def test_trim_then_normalize(self):
        text = "  hello    world  "
        trimmed = trim_lr(text)
        normalized = normalize_ws(trimmed)
        assert normalized == "hello world"

    def test_normalize_then_truncate(self):
        text = "hello    world    foo    bar"
        normalized = normalize_ws(text)
        result, applied = safe_truncate(normalized, 15)
        assert result == "hello world foo…"
        assert applied is True

    def test_context_str_then_normalize(self):
        val = None
        context = to_context_str(val)
        normalized = normalize_ws(context)
        assert normalized == ""

    def test_context_str_with_whitespace_then_normalize(self):
        val = "  hello    world  "
        context = to_context_str(val)
        normalized = normalize_ws(context)
        assert normalized == "hello world"