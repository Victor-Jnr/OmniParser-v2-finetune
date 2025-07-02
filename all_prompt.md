task: 当前 finetune 文件有些不符合要求 它原本是一在线 model florence2 的训练, 但是我现在希望微调它的改版, 当前我并不确认它的数据格式是怎么样的, 我让他人进行了一些推测, 模型文件再
  weights\icon_caption_florence我们应该对它进行微调, 我希望你进一步推测一个兼容微调版本 修改 finetune 文件让它能更好的完成任务   , 在你开启task 之前 你需要读取这些文件 @chat.md @README.md
  @finetune_omniparser_models_fixed.py utils.py demo.py 了解这个项目内容, 更好的工作


------------


如果无法创造更多真实数据，可以考虑数据增强 (Data Augmentation)。例如，对原始图片进行轻微的旋转、裁剪、亮度调整等（注意不要影响图片内容），或者对问题/答案进行同义词替换、句式改写等。-- 这是一个解决当前
  finetune 后模型识别内容依旧相同问题的思路, 现在希望你增加脚本内容, 做出以下处理, 增加 data_impr 3 选项, 对于 指定目录 tranning_data 目录下 florence_format 目录记录的数据进行数据增加,
  对执行坐标位置的图片内容做出之前描述的操作, 后方的数字 3 代表我希望增加的数据倍率, 比如原始数据功 20 个. 最后结果等于 60(我可能会设置更大), 上方有提到多种图片处理方式你应该对这多种进行随机选择后输出,
  比如先随机 旋转, 旋转30度, 再比如随机: 裁剪, 随机到裁剪1%(要在足够小的范围, 否则会导致圆心图片失效,你来决定最大和最小裁剪比例), 最终备份原始 florence_data.json
  写一个新的增强后的数据为florence_data.json, 增强的数据请放到 training_data\florence_format\imgs 下方方便区分增强数据,