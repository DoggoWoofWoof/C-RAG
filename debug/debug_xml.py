import bz2
import xml.etree.ElementTree as ET

path = "data/raw/simplewiki.xml.bz2"
print(f"Reading {path}...")

with bz2.open(path, 'rt', encoding='utf-8') as f:
    context = ET.iterparse(f, events=('start', 'end'))
    for i, (event, elem) in enumerate(context):
        print(f"Event: {event}, Tag: {elem.tag}")
        if i > 50: break
