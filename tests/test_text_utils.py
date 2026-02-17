from functions.utils.text import json_stringify_if_needed, sanitize_psv_value


class TestJsonStringifyIfNeeded:
    """Tests for json_stringify_if_needed."""

    def test_dict_stringified(self):
        data = {"b": 2, "a": 1}
        result = json_stringify_if_needed(data)
        assert isinstance(result, str)
        # sorted keys
        assert result == '{"a": 1, "b": 2}'

    def test_list_stringified(self):
        data = [3, 1, 2]
        result = json_stringify_if_needed(data)
        assert result == "[3, 1, 2]"

    def test_nested_structure(self):
        data = {"a": [1, {"b": 2}]}
        result = json_stringify_if_needed(data)
        assert isinstance(result, str)
        # ensure valid JSON
        import json
        parsed = json.loads(result)
        assert parsed["a"][1]["b"] == 2

    def test_string_passthrough(self):
        assert json_stringify_if_needed("hello") == "hello"

    def test_int_passthrough(self):
        assert json_stringify_if_needed(42) == 42

    def test_none_passthrough(self):
        assert json_stringify_if_needed(None) is None


class TestSanitizePSVValue:
    """Tests for sanitize_psv_value."""

    def test_none_to_empty_string(self):
        assert sanitize_psv_value(None) == ""

    def test_numeric_passthrough(self):
        assert sanitize_psv_value(42) == 42
        assert sanitize_psv_value(3.14) == 3.14
        assert sanitize_psv_value(True) is True

    def test_newline_escaped(self):
        result = sanitize_psv_value("hello\nworld")
        assert result == "hello\\nworld"

    def test_tab_escaped(self):
        result = sanitize_psv_value("hello\tworld")
        assert result == "hello\\tworld"

    def test_crlf_normalized(self):
        result = sanitize_psv_value("hello\r\nworld")
        assert result == "hello\\nworld"

    def test_pipe_delimiter_escaped(self):
        result = sanitize_psv_value("hello|world")
        assert result == "hello\\|world"

    def test_multiple_control_chars(self):
        text = "a\nb\tc|d\r\ne"
        result = sanitize_psv_value(text)
        assert result == "a\\nb\\tc\\|d\\ne"

    def test_idempotency(self):
        text = "hello\nworld"
        once = sanitize_psv_value(text)
        twice = sanitize_psv_value(once)
        assert once == twice  # should not double-escape

    def test_empty_string(self):
        assert sanitize_psv_value("") == ""

    def test_json_string_remains_valid(self):
        """
        Ensure JSON produced by json_stringify_if_needed
        remains valid after PSV sanitization.
        """
        import json

        data = {"a": 1, "b": 2}
        json_str = json_stringify_if_needed(data)
        sanitized = sanitize_psv_value(json_str)

        # Should not break JSON structure
        parsed = json.loads(sanitized)
        assert parsed["a"] == 1
        assert parsed["b"] == 2


class TestPSVIntegrationSafety:
    """Integration tests for PSV behavior."""

    def test_single_line_guarantee(self):
        """
        Sanitized values must never contain real newline characters.
        """
        text = "hello\nworld"
        result = sanitize_psv_value(text)
        assert "\n" not in result
        assert "\r" not in result

    def test_json_with_newlines(self):
        """
        JSON containing newlines should still become single-line safe.
        """
        import json

        data = {"text": "hello\nworld"}
        json_str = json.dumps(data)
        sanitized = sanitize_psv_value(json_str)

        assert "\n" not in sanitized
        parsed = json.loads(sanitized)
        assert parsed["text"] == "hello\nworld"

    def test_determinism(self):
        """
        Same input must always produce identical output.
        """
        text = "a\nb|c\t"
        r1 = sanitize_psv_value(text)
        r2 = sanitize_psv_value(text)
        assert r1 == r2
