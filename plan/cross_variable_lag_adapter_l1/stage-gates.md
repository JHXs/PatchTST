# L1 阶段门

- D0 协议锁定：03/04/05、L0 PASS、正式种子/目录和超参数已记录。
- D1 方法—实验追踪：见 `review/method-experiment-traceability.md`。
- D2 产物契约：见 `tables/table-schema.md` 与 `figures/data-manifest.md`。
- D3 实现验收：P1修复后，参数数、旁路、梯度、F10逐epoch/最佳/最终PatchTST逐元素状态、数据泄漏与输入替换测试通过。
- D4 Smoke 隔离：旧BN污染smoke作废；替代CPU目录带 `_smoke`，单种子、极少预算，汇总器拒绝生成正式 gate。
- D5 双审与验证：py_compile、36项相关unittest、替代CPU smoke与独立CPU复算通过；真实GPU batch512资源smoke因设备未暴露为BLOCKED，D5未关闭。

正式 L1 的性能硬门由 03 第9节定义，本次不判定、不运行也不预览。D5关闭和代码提交后的clean tree是正式启动前置条件。
