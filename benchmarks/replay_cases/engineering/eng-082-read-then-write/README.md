# eng-082-read-then-write

确定性 engineering smoke 用例。生成器：`scripts/generate_replay_case_suite.py`（case_index=82，模式 `transform`）。

工具序列：`<final> -> <tool>{"name": "read_file", "args": {"path": "sample.txt", "start": 1, "end": 20}}</tool> -> <tool name="write_file" path="builds/transformed-082.txt"><content>RECORD-082
Sample RECORD-082
alpha
beta
gamma
</content></tool>`（呈现简写，实际见 fake_outputs.json）。
