"""
Reads all *.fb2 files from the ../data/murakami_fb2s directory and
concats them into one text file, and pushes them to Huggingface Hub
as `vldsavelyev/murakami` dataset repository.
"""

from pathlib import Path
from lxml import etree
import datasets
import fire
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()


# Small chapters are usually the footnotes and the title of the book, skipping by default as it's
# not helping to capture the style of the author anyway.
MIN_CHAPTER_SIZE = 500


def main(
    fb2_dir: Path = Path(
        "/Users/vlad/git/vladsaveliev/huggingface-hub/datasets/vldsavelyev/murakami/data"
    ),
    name: str = "murakami",
):
    chapters_by_title = {}

    fb2s = list(Path(fb2_dir).glob("*.fb2"))
    if len(fb2s) > 0:
        print(f"Found {len(fb2s)} fb2 files in {fb2_dir}")
    else:
        raise ValueError(f"No fb2 files found in {fb2_dir}")

    for bi, path in enumerate(fb2s):
        print(bi, path)

        # Load the FB2 format file
        with path.open("rb") as file:
            fb2_data = file.read()

        # Parse the FB2 format file using lxml
        root = etree.fromstring(fb2_data)

        # Get the title of the book
        title = root.xpath(
            "//fb:title-info/fb:book-title",
            namespaces={"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"},
        )[0].text
        print(title)

        chapters = []

        def _add_chapter(text: str):
            if not text:
                return
            if MIN_CHAPTER_SIZE is not None and len(text) < MIN_CHAPTER_SIZE:
                # print(f"Skipping chapter of length {len(text)}")
                pass
            else:
                # print(f"Adding chapter of length {len(text)}")
                chapters.append(text)

        # All text is stored in <p> tags. There are also <section> tags, which do not have any content,
        # but serve as chapters separators. So we will merge all <p> tags contents between two <section>.
        chapter = ""
        for e in root.iter():
            if e.tag.endswith("}p"):
                chapter += (e.text or "") + (e.tail or "")
            elif e.tag.endswith("}section"):
                _add_chapter(chapter)
                chapter = ""
        _add_chapter(chapter)

        print(f"Found {len(chapters)} chapters")
        # print(f"Chapter sizes: {', '.join(str(len(c)) for c in chapters)}")
        # print()
        chapters_by_title[title] = chapters

    print("Novel by size:")
    for title, chapters in chapters_by_title.items():
        print(f"  {title}: {sum(len(c) for c in chapters):,} characters")


if __name__ == "__main__":
    fire.Fire(main)
