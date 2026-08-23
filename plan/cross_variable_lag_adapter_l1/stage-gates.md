# L1 阶段门

- D0 协议锁定：03/04/05、L0 PASS、正式种子/目录和超参数已记录。
- D1 方法—实验追踪：见 `review/method-experiment-traceability.md`。
- D2 产物契约：见 `tables/table-schema.md` 与 `figures/data-manifest.md`。
- D3 实现验收：P1修复后，参数数、旁路、梯度、F10逐epoch/最佳/最终PatchTST逐元素状态、数据泄漏与输入替换测试通过。
- D4 Smoke 隔离：旧BN污染smoke作废；替代CPU目录带 `_smoke`，单种子、极少预算，汇总器拒绝生成正式 gate。
- D5 双审与验证：py_compile、36项相关unittest、替代CPU smoke、真实GPU batch512资源smoke与严格联合复算全部通过，D5已关闭。

- D6 正式性能门：2060–2062三种子选择已运行，独立完整性PASS，但F10、B1、B2、B-flat与M-shuffle硬门失败，状态STOP。

正式L1已在clean tree上完成。根据预注册规则，D6 STOP终止本路线；不得进入confirmation、打开正式test或回调配置。
