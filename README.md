1 本项目先进行人脸检测，使用的是dlib工具，可以检测出68个人脸关键点。
2 本项目实现了人脸的裁剪，根据人脸关键点进行瘦脸，大眼操作。
3 人脸对齐原理：先进行人脸区域的裁剪，然后进行关键点检测，接着利用两眼之间的连线与水平的夹角进行仿射变换进行人脸的矫正。
4 瘦脸原理，参考
https://github.com/xlc-github/facetransform/blob/master/img/%E7%98%A6%E8%84%B8%E5%85%AC%E5%BC%8F.png
