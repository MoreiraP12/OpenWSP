"""shakespearelang 1.0.0 reads only digits for 'Listen to your heart!', so a negative input number fails ('No numeric
input was given'). SPL integers are signed and the polyglot tasks have negative inputs: accept one leading - or +."""
import pathlib
import shakespearelang

p = pathlib.Path(shakespearelang.__file__).parent / "_input.py"
s = p.read_text()
old = '        number_input = ""\n        while self._input_buffer and self._input_buffer[0].isdigit():'
new = ('        number_input = ""\n'
       '        if self._input_buffer[:1] in ("-", "+") and self._input_buffer[1:2].isdigit():\n'
       '            number_input, self._input_buffer = self._input_buffer[0], self._input_buffer[1:]\n'
       '        while self._input_buffer and self._input_buffer[0].isdigit():')
assert s.count(old) == 1, "shakespearelang input code changed"
p.write_text(s.replace(old, new))
