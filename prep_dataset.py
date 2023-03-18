"""
Reads all *.fb2 files from the ../data/murakami_fb2s directory and concatenates
them into one file ../data/murakami.txt
"""

from pathlib import Path
from lxml import etree

# Number of initial <p> element to take from each fb2, by number. This allows to skip 
# intros and other junk in the beginning of an fb2. This is built semi-manually using 
# the `helper_to_find_first_paragraphs` func.
START_PARAGRAPHS = {
    3: 5,
    6: 27,
    7: 3,
    9: 4,
    10: 3,
    12: 11,
    18: 5,
    20: 3,
    21: 5,
}


def helper_to_find_first_paragraphs(paragraphs, title, book_number, n=30):
    """
    Helps to eyeball first few paragraphs of a book to skip junk paragraphs
    in the beginning and manually construct the `tart_paragraphs` dict.
    """
    found_paragraphs = []
    skipping = True
    for i, p in enumerate(list(paragraphs)[:n]):
        if p.text is None:
            continue
        if book_number in START_PARAGRAPHS and i >= START_PARAGRAPHS[book_number]:
            skipping = False
        if skipping and p.text.lower() == title.lower():
            skipping = False
        if not skipping:
            found_paragraphs.append(f"   {i} {p.text}")

    if found_paragraphs:
        print("✅")
        print("\n".join(found_paragraphs))

    else:
        print("❌")
        for i, p in enumerate(list(paragraphs)[:30]):
            print(f"   {i} {p.text}")


def main():
    basedir = Path("/Users/vlad/googledrive/AI/datasets/murakami")

    text_by_name = {}

    for bi, path in enumerate((basedir / "fb2").glob("*.fb2")):
        print(bi, path)

        # Load the FB2 format file
        with path.open("rb") as file:
            fb2_data = file.read()

        # Print structure of the FB2 format file
        # print(etree.tostring(etree.fromstring(fb2_data), pretty_print=True))

        # Parse the FB2 format file using lxml
        root = etree.fromstring(fb2_data)

        # Print the title of the book
        title = root.xpath(
            "//fb:title-info/fb:book-title",
            namespaces={"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"},
        )[0].text
        print(title)

        paragraphs = root.xpath(
            "//fb:p",
            namespaces={"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"},
        )

        # UNCOMMENT THIS TO BUILD `START_PARAGRAPHS`
        # helper_to_find_first_paragraphs(paragraphs, title, bi)

        found_paragraphs = []
        skipping = True
        for pi, p in enumerate(paragraphs):
            if p.text is None:
                continue
            if bi in START_PARAGRAPHS and pi >= START_PARAGRAPHS[bi]:
                skipping = False
            if skipping and p.text.lower() == title.lower():
                skipping = False
            if not skipping:
                found_paragraphs.append(p)
        print(f"Found {len(found_paragraphs)} paragraphs")

        text_by_name[title] = ""
        for p in found_paragraphs:
            text_by_name[title] += p.text.replace(" ", " ") + "\n"
        text_by_name[title] += "\n"

    assert text_by_name

    print("Novel by size:")
    for title, text in text_by_name.items():
        print(f"  {title}: {len(text):,} characters")

    smallest_title = min(text_by_name, key=lambda k: len(text_by_name[k]))
    print(
        f"Using smallest novel {smallest_title} "
        f"({len(text_by_name[smallest_title]):,} characters) as a test set"
    )

    with open(basedir / "murakami_train.txt", "w") as f:
        for title, text in text_by_name.items():
            if title != smallest_title:
                f.write(text)

    with open(basedir / "murakami_test.txt", "w") as f:
        f.write(text_by_name[smallest_title])


if __name__ == "__main__":
    main()
