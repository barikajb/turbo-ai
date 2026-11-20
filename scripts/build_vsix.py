"""Package vscode-extension/ into a .vsix without Node or vsce.

A .vsix is a zip with a manifest. Usage:
    python scripts/build_vsix.py
    code --install-extension dist/turbo-ai-local-0.1.0.vsix
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
EXT_DIR = ROOT / "vscode-extension"
FILES = ["package.json", "extension.js", "lib.js", "README.md"]

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json"/>
  <Default Extension=".js" ContentType="application/javascript"/>
  <Default Extension=".md" ContentType="text/markdown"/>
  <Default Extension=".vsixmanifest" ContentType="text/xml"/>
</Types>
"""

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{name}" Version="{version}" Publisher="{publisher}"/>
    <DisplayName>{display_name}</DisplayName>
    <Description xml:space="preserve">{description}</Description>
    <Categories>{categories}</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{engine}"/>
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value=""/>
      <Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value=""/>
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="workspace"/>
      <Property Id="Microsoft.VisualStudio.Code.LocalizedLanguages" Value=""/>
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code"/>
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true"/>
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details" Path="extension/README.md" Addressable="true"/>
  </Assets>
</PackageManifest>
"""


def build() -> Path:
    package = json.loads((EXT_DIR / "package.json").read_text(encoding="utf-8"))
    manifest = MANIFEST.format(
        name=escape(package["name"]),
        version=escape(package["version"]),
        publisher=escape(package["publisher"]),
        display_name=escape(package.get("displayName", package["name"])),
        description=escape(package.get("description", "")),
        categories=escape(",".join(package.get("categories", []))),
        engine=escape(package["engines"]["vscode"]),
    )

    out = ROOT / "dist" / f"{package['name']}-{package['version']}.vsix"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", CONTENT_TYPES)
        archive.writestr("extension.vsixmanifest", manifest)
        for name in FILES:
            source = EXT_DIR / name
            if not source.is_file():
                sys.exit(f"Missing {source}")
            archive.write(source, f"extension/{name}")
    return out


if __name__ == "__main__":
    print(build())
