#!/bin/bash
# SoftPaw UST — Dataset Download Script
#
# Downloads all required training datasets:
#   1. MathWriting (Google) — 630K math expressions with stroke data
#   2. IAM Online Handwriting — requires manual registration
#   3. QuickDraw (Google) — simplified drawings
#
# Usage: bash scripts/download_data.sh [output_dir]

set -e

OUTPUT_DIR="${1:-./datasets}"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "SoftPaw UST — Dataset Download"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. MathWriting
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] MathWriting Dataset"
echo "  Source: https://github.com/google-research/google-research/tree/master/mathwriting"

MATHWRITING_DIR="$OUTPUT_DIR/mathwriting"
if [ -d "$MATHWRITING_DIR" ] && [ "$(ls -A $MATHWRITING_DIR 2>/dev/null)" ]; then
    echo "  Already exists at $MATHWRITING_DIR, skipping."
else
    mkdir -p "$MATHWRITING_DIR"
    echo "  Downloading MathWriting dataset..."
    echo "  NOTE: The MathWriting dataset needs to be downloaded from Google Cloud Storage."
    echo "  Visit: https://github.com/google-research/google-research/tree/master/mathwriting"
    echo "  Download the InkML files and extract to: $MATHWRITING_DIR"
    echo ""
    echo "  Expected structure:"
    echo "    $MATHWRITING_DIR/train/*.inkml"
    echo "    $MATHWRITING_DIR/val/*.inkml"
    echo "    $MATHWRITING_DIR/test/*.inkml"
    echo ""
    echo "  If available via gsutil:"
    echo "    gsutil -m cp -r gs://mathwriting_data/* $MATHWRITING_DIR/"
fi

# ---------------------------------------------------------------------------
# 2. IAM Online Handwriting Database
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] IAM Online Handwriting Database"
echo "  Source: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database"

IAM_DIR="$OUTPUT_DIR/iam_online"
if [ -d "$IAM_DIR" ] && [ "$(ls -A $IAM_DIR 2>/dev/null)" ]; then
    echo "  Already exists at $IAM_DIR, skipping."
else
    mkdir -p "$IAM_DIR"
    echo "  MANUAL DOWNLOAD REQUIRED"
    echo "  1. Register at: https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database"
    echo "  2. Download the 'lineStrokes' dataset (XML files with stroke data)"
    echo "  3. Extract to: $IAM_DIR/"
    echo ""
    echo "  Expected structure:"
    echo "    $IAM_DIR/lineStrokes/**/*.xml"
    echo "  or"
    echo "    $IAM_DIR/train/*.xml"
    echo "    $IAM_DIR/val/*.xml"
    echo "    $IAM_DIR/test/*.xml"
fi

# ---------------------------------------------------------------------------
# 3. QuickDraw (Simplified)
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Google QuickDraw Dataset (Simplified)"
echo "  Source: https://github.com/googlecreativelab/quickdraw-dataset"

QUICKDRAW_DIR="$OUTPUT_DIR/quickdraw"
if [ -d "$QUICKDRAW_DIR" ] && [ "$(ls -A $QUICKDRAW_DIR 2>/dev/null)" ]; then
    echo "  Already exists at $QUICKDRAW_DIR, skipping."
else
    mkdir -p "$QUICKDRAW_DIR"
    echo "  Downloading QuickDraw simplified NDJSON files..."
    echo "  Selecting 100 common categories..."

    # 100 diverse categories for drawing variety
    CATEGORIES=(
        "airplane" "alarm_clock" "apple" "axe" "backpack"
        "banana" "baseball_bat" "basketball" "bear" "bed"
        "bicycle" "bird" "book" "bowtie" "brain"
        "bridge" "bucket" "bus" "butterfly" "cake"
        "camera" "candle" "car" "castle" "cat"
        "chair" "church" "circle" "clock" "cloud"
        "coffee_cup" "compass" "computer" "cookie" "couch"
        "cow" "crown" "cup" "diamond" "dog"
        "door" "dragon" "ear" "elephant" "envelope"
        "eye" "face" "fan" "fire" "fish"
        "flower" "fork" "frog" "guitar" "hamburger"
        "hand" "hat" "heart" "helicopter" "horse"
        "hospital" "house" "ice_cream" "key" "knife"
        "laptop" "leaf" "light_bulb" "lightning" "lion"
        "moon" "mountain" "mushroom" "ocean" "octopus"
        "paintbrush" "palm_tree" "panda" "pants" "parachute"
        "pencil" "penguin" "piano" "pig" "pizza"
        "rabbit" "rainbow" "robot" "rocket" "sailboat"
        "scissors" "shark" "shoe" "skull" "smiley_face"
        "snake" "spider" "star" "strawberry" "sun"
        "sword" "table" "telephone" "tree" "umbrella"
    )

    BASE_URL="https://storage.googleapis.com/quickdraw_dataset/full/simplified"

    for cat in "${CATEGORIES[@]}"; do
        DEST="$QUICKDRAW_DIR/${cat}.ndjson"
        if [ -f "$DEST" ]; then
            continue
        fi
        # URL-encode the category name (spaces → %20)
        URL_CAT=$(echo "$cat" | sed 's/ /%20/g')
        echo "  Downloading: $cat"
        curl -sL "${BASE_URL}/${URL_CAT}.ndjson" -o "$DEST" 2>/dev/null || \
            echo "    Warning: Failed to download $cat"
    done

    echo "  Downloaded $(ls $QUICKDRAW_DIR/*.ndjson 2>/dev/null | wc -l) categories"
fi

# ---------------------------------------------------------------------------
# 4. Handwriting Fonts (for synthetic text generation)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Handwriting Fonts (for synthetic handwriting generation)"

FONT_DIR="$OUTPUT_DIR/fonts/handwriting"
if [ -d "$FONT_DIR" ] && [ "$(ls $FONT_DIR/*.ttf $FONT_DIR/*.otf 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "  Already have fonts in $FONT_DIR, skipping."
else
    mkdir -p "$FONT_DIR"
    echo "  Downloading free handwriting fonts from Google Fonts..."

    # Google Fonts handwriting fonts (all open source, SIL OFL license)
    FONT_FAMILIES=(
        "Caveat"
        "Dancing+Script"
        "Indie+Flower"
        "Patrick+Hand"
        "Shadows+Into+Light"
        "Architects+Daughter"
        "Coming+Soon"
        "Covered+By+Your+Grace"
        "Gloria+Hallelujah"
        "Handlee"
        "Just+Another+Hand"
        "Kalam"
        "Kristi"
        "La+Belle+Aurore"
        "Nothing+You+Could+Do"
        "Over+the+Rainbow"
        "Reenie+Beanie"
        "Rock+Salt"
        "Sacramento"
        "Satisfy"
        "Schoolbell"
        "Short+Stack"
        "Sue+Ellen+Francisco"
        "Swanky+and+Moo+Moo"
        "Waiting+for+the+Sunrise"
        "Yellowtail"
        "Zeyada"
        "Permanent+Marker"
        "Amatic+SC"
        "Homemade+Apple"
    )

    for family in "${FONT_FAMILIES[@]}"; do
        CLEAN_NAME=$(echo "$family" | sed 's/+/_/g')
        echo "  Downloading: $CLEAN_NAME"
        # Google Fonts API provides zip downloads
        curl -sL "https://fonts.google.com/download?family=${family}" \
            -o "/tmp/${CLEAN_NAME}.zip" 2>/dev/null && \
        unzip -qo "/tmp/${CLEAN_NAME}.zip" -d "/tmp/${CLEAN_NAME}_extracted" 2>/dev/null && \
        find "/tmp/${CLEAN_NAME}_extracted" -name "*.ttf" -exec cp {} "$FONT_DIR/" \; 2>/dev/null && \
        find "/tmp/${CLEAN_NAME}_extracted" -name "*.otf" -exec cp {} "$FONT_DIR/" \; 2>/dev/null && \
        rm -rf "/tmp/${CLEAN_NAME}.zip" "/tmp/${CLEAN_NAME}_extracted" || \
            echo "    Warning: Failed to download $CLEAN_NAME"
    done

    FONT_COUNT=$(ls "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l)
    echo "  Downloaded $FONT_COUNT font files"
    echo ""
    echo "  Want more variety? Add any .ttf/.otf handwriting fonts to: $FONT_DIR/"
    echo "  Good sources: dafont.com (handwriting category), fontsquirrel.com"
fi

# ---------------------------------------------------------------------------
# 5. Text Corpus (for synthetic handwriting vocabulary diversity)
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] English Text Corpus"

CORPUS_DIR="$OUTPUT_DIR/fonts"
CORPUS_FILE="$CORPUS_DIR/corpus.txt"
if [ -f "$CORPUS_FILE" ] && [ "$(wc -l < "$CORPUS_FILE")" -gt 1000 ]; then
    echo "  Corpus already exists at $CORPUS_FILE ($(wc -l < "$CORPUS_FILE") lines), skipping."
else
    mkdir -p "$CORPUS_DIR"
    echo "  Generating diverse English text corpus..."

    # Wikitext-103 raw (public domain, diverse English text)
    echo "  Downloading Wikitext-103 raw text..."
    WIKI_URL="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    if curl -sL "$WIKI_URL" -o "/tmp/wikitext103.zip" 2>/dev/null; then
        unzip -qo "/tmp/wikitext103.zip" -d "/tmp/wikitext103" 2>/dev/null
        # Extract clean sentences (skip headers, empty lines, short lines)
        if [ -f "/tmp/wikitext103/wikitext-103-raw/wiki.train.raw" ]; then
            grep -v '^$' "/tmp/wikitext103/wikitext-103-raw/wiki.train.raw" | \
            grep -v '^ = ' | \
            grep -v '^ ==' | \
            awk 'length > 10 && length < 150' | \
            shuf | head -100000 > "$CORPUS_FILE"
            echo "  Extracted $(wc -l < "$CORPUS_FILE") sentences from Wikitext-103"
        fi
        rm -rf "/tmp/wikitext103.zip" "/tmp/wikitext103"
    else
        echo "  Warning: Could not download Wikitext-103."
        echo "  The synthetic generator will still work with built-in word banks."
        echo "  For best results, place a corpus.txt file (one sentence per line) at:"
        echo "    $CORPUS_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Dataset Download Summary"
echo "============================================"
echo ""

# Count files
MW_COUNT=$(find "$MATHWRITING_DIR" -name "*.inkml" 2>/dev/null | wc -l)
IAM_COUNT=$(find "$IAM_DIR" -name "*.xml" 2>/dev/null | wc -l)
QD_COUNT=$(ls "$QUICKDRAW_DIR"/*.ndjson 2>/dev/null | wc -l)
FONT_COUNT=$(ls "$FONT_DIR"/*.ttf "$FONT_DIR"/*.otf 2>/dev/null | wc -l)
CORPUS_LINES=$(wc -l < "$CORPUS_FILE" 2>/dev/null || echo "0")

echo "MathWriting:  $MW_COUNT InkML files (math with strokes)"
echo "IAM Online:   $IAM_COUNT XML files (real handwriting)"
echo "QuickDraw:    $QD_COUNT NDJSON categories (drawings)"
echo "Fonts:        $FONT_COUNT handwriting fonts (for synthetic text)"
echo "Corpus:       $CORPUS_LINES sentences (vocabulary diversity)"
echo ""

if [ "$MW_COUNT" -eq 0 ] || [ "$IAM_COUNT" -eq 0 ]; then
    echo "WARNING: Some datasets need manual download."
    echo "See instructions above for MathWriting and IAM Online."
    echo ""
fi

if [ "$FONT_COUNT" -eq 0 ]; then
    echo "WARNING: No handwriting fonts found."
    echo "Without fonts, you'll only have IAM's 13K text samples."
    echo "With fonts, you get 500K+ synthetic text samples."
    echo ""
fi

echo "Data composition:"
echo "  Text:     ~500K synthetic (${FONT_COUNT} fonts × ${CORPUS_LINES}+ vocab) + ${IAM_COUNT} real"
echo "  Math:     ${MW_COUNT} real math expressions from MathWriting"
echo "  Drawings: ${QD_COUNT} categories × 20K samples from QuickDraw"
echo "  Gestures: generated on-the-fly (unlimited)"
echo ""
echo "Ready for training: python train.py --data_dir $OUTPUT_DIR"
