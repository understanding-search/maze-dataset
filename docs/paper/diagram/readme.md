- `main.svg` contains references to `ascii.svg`, `mazeplot.svg`, `pixels.svg`. `diagram.svg` has these embedded directly.
- `tokens.html` is not referenced since i don't think there is a way to do this -- we just embed it directly. the actual version also doesn't color the text background but just makes boxes
- to export to pdf, firefox for some reason breaks links. I tried vivaldi and that worked, so probably a chromium browser will work.
- `pdfcrop` somehow stops the links from working, so we print to a legal sized paper in landscape. this leaves a bit of whitespace at the bottom, but it's close enough. lets hope including it in latex doesn't break things.
- aaaand including it in latex breaks things. see `diagram.tikz` for manually overlaying links on the pdf