# Package

version       = "0.1.0"
author        = "Double_oxygeN"
description   = "neural network sample 1"
license       = "Apache-2.0"

srcDir        = "src"
binDir        = "bin"
bin           = @["neural1"]
skipDirs      = @["tests"]

# Dependencies

requires "nim >= 0.17.2"

# tasks

task run, "run the project":
  exec "nimble build -w:on --colors:on && ./bin/neural1"

task release, "do release build":
  exec "nimble build -d:release --opt:speed && strip ./bin/neural1"

task cleanup, "clean up files":
  exec "rm -f bin/* && rm -rf src/nimcache"
  exec "find tests -type f ! -name \"*.*\" -delete && rm -rf tests/nimcache"
  exec "rm -f docs/*.html"

task docgen, "generate documentation":
  exec "nimble doc2 --project -o:docs src/neural1pkg/neuralnet.nim"
