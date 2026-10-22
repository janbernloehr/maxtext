# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference test collection must tolerate absent optional PyTorch dependencies."""
import ast
from pathlib import Path
import sys


def test_reference_setup_without_torch(monkeypatch):
  """Exercise collection setup without loading JAX or downloading publisher code."""
  source_path = Path(__file__).with_name("ming_layers_test.py")
  source = ast.parse(source_path.read_text(encoding="utf-8"))
  prefix = []
  for statement in source.body:
    if isinstance(statement, ast.ImportFrom) and statement.module == "flax":
      break
    prefix.append(statement)
  monkeypatch.setitem(sys.modules, "torch", None)
  namespace = {"__name__": "ming_reference_setup"}
  exec(compile(ast.Module(body=prefix, type_ignores=[]), str(source_path), "exec"), namespace)  # pylint: disable=exec-used
  assert namespace["HAS_TORCH"] is False
  assert namespace["HAS_REF"] is False
