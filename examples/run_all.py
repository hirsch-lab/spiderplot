#!/usr/bin/env python
import example1
import example2
import example3
import example4

outdir = "out/"
print("Find output under '%s'" % outdir)
print("Running example1...")
example1.run(save=True, interactive=False, outdir=outdir)
print("Running example2...")
example2.run(save=True, interactive=False, outdir=outdir)
print("Running example3...")
example3.run(save=True, interactive=False, outdir=outdir)
print("Running example4...")
example4.run(save=True, interactive=False, outdir=outdir)
