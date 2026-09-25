"""op - MAGMa 优化版本

v8版本: 分数延迟计算优化
- 碎片生成时不计算分数
- 只对匹配到的碎片计算分数
- 大幅减少 score_fragment 调用次数
"""

from .fragmentation_op_v8 import FragmentEngine

__all__ = ['FragmentEngine']
