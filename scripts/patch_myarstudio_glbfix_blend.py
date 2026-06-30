#!/usr/bin/env python3
"""Patch the embedded My AR Studio GLB Fix script for newer Blender versions."""

from __future__ import annotations

import pathlib
import re
import sys

import bpy


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise RuntimeError(f"expected snippet not found: {old!r}")
    return source.replace(old, new, 1)


def main() -> int:
    if "--" not in sys.argv:
        raise RuntimeError("usage: blender --background --python patch_myarstudio_glbfix_blend.py -- file.blend")
    blend_path = pathlib.Path(sys.argv[sys.argv.index("--") + 1]).resolve()

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    text = bpy.data.texts.get("GLBfix_v05.py")
    if text is None:
        raise RuntimeError("GLBfix_v05.py text block not found")

    source = text.as_string()
    helper = (
        "\n\ndef blender_at_least(major, minor=0):\n"
        "    return (BV[0] > major) or ((BV[0] == major) and (BV[1] >= minor))\n"
    )
    source = re.sub(
        r"\n\ndef blender_at_least\(major, minor=0\):\n"
        r"    return \(BV\[0\] > major\) or \(\(BV\[0\] == major\) and \(BV\[1\] >= minor\)\)\n",
        "",
        source,
    )
    source = replace_once(source, "BV = bpy.app.version\n", "BV = bpy.app.version" + helper + "\n")
    source = source.replace(
        "if (BV[0] < 4) or ((BV[0] == 4) and (BV[1] < 2)) :",
        "if not blender_at_least(4, 2):",
    )
    source = source.replace(
        "if (BV[0] >= 4) and (BV[1] >= 2):",
        "if blender_at_least(4, 2):",
    )
    source = source.replace(
        "if (BV[0] >= 4) and (BV[1] >= 2): ",
        "if blender_at_least(4, 2): ",
    )
    source = source.replace("bpy.ops.export_Scene.gltf", "bpy.ops.export_scene.gltf")
    source = re.sub(
        r"\n        (?:if bpy\.context\.area and bpy\.context\.area\.type == 'VIEW_3D':\n)+"
        r"\s*bpy\.ops\.view3d\.view_all\(use_all_regions=True, center=True\)\n",
        "\n        if bpy.context.area and bpy.context.area.type == 'VIEW_3D':\n"
        "            bpy.ops.view3d.view_all(use_all_regions=True, center=True)\n",
        source,
    )
    source = source.replace(
        "\n        bpy.ops.view3d.view_all(use_all_regions=True, center=True)\n",
        "\n        if bpy.context.area and bpy.context.area.type == 'VIEW_3D':\n"
        "            bpy.ops.view3d.view_all(use_all_regions=True, center=True)\n",
    )
    source = re.sub(
        r"\n\s*if bpy\.context\.area and bpy\.context\.area\.type == 'VIEW_3D':\n"
        r"\s*if bpy\.context\.area and bpy\.context\.area\.type == 'VIEW_3D':\n"
        r"\s*bpy\.ops\.view3d\.view_all\(use_all_regions=True, center=True\)\n",
        "\n        if bpy.context.area and bpy.context.area.type == 'VIEW_3D':\n"
        "            bpy.ops.view3d.view_all(use_all_regions=True, center=True)\n",
        source,
    )

    text.clear()
    text.write(source)
    text.use_module = True

    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    print(f"patched: {blend_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
