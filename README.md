# repo2

## 引用 result/task1_img 中的图片

下面会引用 `result/task1_img` 目录下的所有图片文件（支持常见扩展名：png、jpg、jpeg、gif）。如果该目录中存在图片，它们将会在此处渲染。当前为占位引用；当实际文件添加到目录后，图片会在此处显示。

以下为占位示例（当对应文件存在时即会显示）：

![result/task1_img - Image](result/task1_img/gray.png)
![result/task1_img - Image](result/task1_img/hsv.png)
![result/task1_img - Image](result/task1_img/mean_filter.png)
![result/task1_img - Image](result/task1_img/gaussian_filter.png)
![result/task1_img - Image](result/task1_img/median_filter.png)
![result/task1_img - Image](result/task1_img/open.png)
![result/task1_img - Image](result/task1_img/close.png)
![result/task1_img - Image](result/task1_img/red_mask.png)
![result/task1_img - Image](result/task1_img/contours_boxes.png)


如果想自动将目录中所有图片插入 README，可以在仓库根目录运行下面的一行命令来生成可复制到本文件的 Markdown 列表：

```bash
# 在仓库根目录运行：
find result/task1_img -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.gif" \) -printf "![%f](result/task1_img/%f)\n"
```

将上述命令的输出粘贴到本节即可批量引用目录中所有图片。
