from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Union
from .xlink import (
    ActuateValue,
    ShowValue,
    TypeValue,
)
from .xml import (
    LangValue,
    SpaceValue,
)


class AboveBelow(Enum):
    """
    The above-below type is used to indicate whether one element appears above
    or below another element.
    """
    ABOVE = "above"
    BELOW = "below"


class AccidentalValue(Enum):
    """The accidental-value type represents notated accidentals supported by
    MusicXML.

    In the MusicXML 2.0 DTD this was a string with values that could be
    included. The XSD strengthens the data typing to an enumerated list.
    The quarter- and three-quarters- accidentals are Tartini-style
    quarter-tone accidentals. The -down and -up accidentals are quarter-
    tone accidentals that include arrows pointing down or up. The slash-
    accidentals are used in Turkish classical music. The numbered sharp
    and flat accidentals are superscripted versions of the accidental
    signs, used in Turkish folk music. The sori and koron accidentals
    are microtonal sharp and flat accidentals used in Iranian and
    Persian music. The other accidental covers accidentals other than
    those listed here. It is usually used in combination with the smufl
    attribute to specify a particular SMuFL accidental. The smufl
    attribute may be used with any accidental value to help specify the
    appearance of symbols that share the same MusicXML semantics.
    """
    SHARP = "sharp"
    NATURAL = "natural"
    FLAT = "flat"
    DOUBLE_SHARP = "double-sharp"
    SHARP_SHARP = "sharp-sharp"
    FLAT_FLAT = "flat-flat"
    NATURAL_SHARP = "natural-sharp"
    NATURAL_FLAT = "natural-flat"
    QUARTER_FLAT = "quarter-flat"
    QUARTER_SHARP = "quarter-sharp"
    THREE_QUARTERS_FLAT = "three-quarters-flat"
    THREE_QUARTERS_SHARP = "three-quarters-sharp"
    SHARP_DOWN = "sharp-down"
    SHARP_UP = "sharp-up"
    NATURAL_DOWN = "natural-down"
    NATURAL_UP = "natural-up"
    FLAT_DOWN = "flat-down"
    FLAT_UP = "flat-up"
    DOUBLE_SHARP_DOWN = "double-sharp-down"
    DOUBLE_SHARP_UP = "double-sharp-up"
    FLAT_FLAT_DOWN = "flat-flat-down"
    FLAT_FLAT_UP = "flat-flat-up"
    ARROW_DOWN = "arrow-down"
    ARROW_UP = "arrow-up"
    TRIPLE_SHARP = "triple-sharp"
    TRIPLE_FLAT = "triple-flat"
    SLASH_QUARTER_SHARP = "slash-quarter-sharp"
    SLASH_SHARP = "slash-sharp"
    SLASH_FLAT = "slash-flat"
    DOUBLE_SLASH_FLAT = "double-slash-flat"
    SHARP_1 = "sharp-1"
    SHARP_2 = "sharp-2"
    SHARP_3 = "sharp-3"
    SHARP_5 = "sharp-5"
    FLAT_1 = "flat-1"
    FLAT_2 = "flat-2"
    FLAT_3 = "flat-3"
    FLAT_4 = "flat-4"
    SORI = "sori"
    KORON = "koron"
    OTHER = "other"


class ArrowDirection(Enum):
    """
    The arrow-direction type represents the direction in which an arrow points,
    using Unicode arrow terminology.
    """
    LEFT = "left"
    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    NORTHWEST = "northwest"
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"
    LEFT_RIGHT = "left right"
    UP_DOWN = "up down"
    NORTHWEST_SOUTHEAST = "northwest southeast"
    NORTHEAST_SOUTHWEST = "northeast southwest"
    OTHER = "other"


class ArrowStyle(Enum):
    """The arrow-style type represents the style of an arrow, using Unicode
    arrow terminology.

    Filled and hollow arrows indicate polygonal single arrows. Paired
    arrows are duplicate single arrows in the same direction. Combined
    arrows apply to double direction arrows like left right, indicating
    that an arrow in one direction should be combined with an arrow in
    the other direction.
    """
    SINGLE = "single"
    DOUBLE = "double"
    FILLED = "filled"
    HOLLOW = "hollow"
    PAIRED = "paired"
    COMBINED = "combined"
    OTHER = "other"


class BackwardForward(Enum):
    """The backward-forward type is used to specify repeat directions.

    The start of the repeat has a forward direction while the end of the
    repeat has a backward direction.
    """
    BACKWARD = "backward"
    FORWARD = "forward"


class BarStyle(Enum):
    """The bar-style type represents barline style information.

    Choices are regular, dotted, dashed, heavy, light-light, light-
    heavy, heavy-light, heavy-heavy, tick (a short stroke through the
    top line), short (a partial barline between the 2nd and 4th lines),
    and none.
    """
    REGULAR = "regular"
    DOTTED = "dotted"
    DASHED = "dashed"
    HEAVY = "heavy"
    LIGHT_LIGHT = "light-light"
    LIGHT_HEAVY = "light-heavy"
    HEAVY_LIGHT = "heavy-light"
    HEAVY_HEAVY = "heavy-heavy"
    TICK = "tick"
    SHORT = "short"
    NONE = "none"


class BeamValue(Enum):
    """
    The beam-value type represents the type of beam associated with each of 8
    beam levels (up to 1024th notes) available for each note.
    """
    BEGIN = "begin"
    CONTINUE = "continue"
    END = "end"
    FORWARD_HOOK = "forward hook"
    BACKWARD_HOOK = "backward hook"


class BeaterValue(Enum):
    """The beater-value type represents pictograms for beaters, mallets, and
    sticks that do not have different materials represented in the pictogram.

    The finger and hammer values are in addition to Stone's list.
    """
    BOW = "bow"
    CHIME_HAMMER = "chime hammer"
    COIN = "coin"
    DRUM_STICK = "drum stick"
    FINGER = "finger"
    FINGERNAIL = "fingernail"
    FIST = "fist"
    GUIRO_SCRAPER = "guiro scraper"
    HAMMER = "hammer"
    HAND = "hand"
    JAZZ_STICK = "jazz stick"
    KNITTING_NEEDLE = "knitting needle"
    METAL_HAMMER = "metal hammer"
    SLIDE_BRUSH_ON_GONG = "slide brush on gong"
    SNARE_STICK = "snare stick"
    SPOON_MALLET = "spoon mallet"
    SUPERBALL = "superball"
    TRIANGLE_BEATER = "triangle beater"
    TRIANGLE_BEATER_PLAIN = "triangle beater plain"
    WIRE_BRUSH = "wire brush"


class BendShape(Enum):
    """
    The bend-shape type distinguishes between the angled bend symbols commonly
    used in standard notation and the curved bend symbols commonly used in both
    tablature and standard notation.
    """
    ANGLED = "angled"
    CURVED = "curved"


@dataclass
class Bookmark:
    """
    The bookmark type serves as a well-defined target for an incoming simple
    XLink.
    """
    class Meta:
        name = "bookmark"

    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    element: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    position: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


class BreathMarkValue(Enum):
    """
    The breath-mark-value type represents the symbol used for a breath mark.
    """
    VALUE = ""
    COMMA = "comma"
    TICK = "tick"
    UPBOW = "upbow"
    SALZEDO = "salzedo"


class CaesuraValue(Enum):
    """
    The caesura-value type represents the shape of the caesura sign.
    """
    NORMAL = "normal"
    THICK = "thick"
    SHORT = "short"
    CURVED = "curved"
    SINGLE = "single"
    VALUE = ""


class CancelLocation(Enum):
    """The cancel-location type is used to indicate where a key signature
    cancellation appears relative to a new key signature: to the left, to the
    right, or before the barline and to the left.

    It is left by default. For mid-measure key elements, a cancel-
    location of before-barline should be treated like a cancel-location
    of left.
    """
    LEFT = "left"
    RIGHT = "right"
    BEFORE_BARLINE = "before-barline"


class CircularArrow(Enum):
    """
    The circular-arrow type represents the direction in which a circular arrow
    points, using Unicode arrow terminology.
    """
    CLOCKWISE = "clockwise"
    ANTICLOCKWISE = "anticlockwise"


class ClefSign(Enum):
    """The clef-sign type represents the different clef symbols.

    The jianpu sign indicates that the music that follows should be in
    jianpu numbered notation, just as the TAB sign indicates that the
    music that follows should be in tablature notation. Unlike TAB, a
    jianpu sign does not correspond to a visual clef notation. The none
    sign is deprecated as of MusicXML 4.0. Use the clef element's print-
    object attribute instead. When the none sign is used, notes should
    be displayed as if in treble clef.
    """
    G = "G"
    F = "F"
    C = "C"
    PERCUSSION = "percussion"
    TAB = "TAB"
    JIANPU = "jianpu"
    NONE = "none"


class CssFontSize(Enum):
    """
    The css-font-size type includes the CSS font sizes used as an alternative
    to a numeric point size.
    """
    XX_SMALL = "xx-small"
    X_SMALL = "x-small"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    X_LARGE = "x-large"
    XX_LARGE = "xx-large"


class DegreeSymbolValue(Enum):
    """
    The degree-symbol-value type indicates which symbol should be used in
    specifying a degree.
    """
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"
    HALF_DIMINISHED = "half-diminished"


class DegreeTypeValue(Enum):
    """
    The degree-type-value type indicates whether the current degree element is
    an addition, alteration, or subtraction to the kind of the current chord in
    the harmony element.
    """
    ADD = "add"
    ALTER = "alter"
    SUBTRACT = "subtract"


@dataclass
class Distance:
    """The distance element represents standard distances between notation
    elements in tenths.

    The type attribute defines what type of distance is being defined.
    Valid values include hyphen (for hyphens in lyrics) and beam.
    """
    class Meta:
        name = "distance"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class EffectValue(Enum):
    """The effect-value type represents pictograms for sound effect percussion
    instruments.

    The cannon, lotus flute, and megaphone values are in addition to
    Stone's list.
    """
    ANVIL = "anvil"
    AUTO_HORN = "auto horn"
    BIRD_WHISTLE = "bird whistle"
    CANNON = "cannon"
    DUCK_CALL = "duck call"
    GUN_SHOT = "gun shot"
    KLAXON_HORN = "klaxon horn"
    LIONS_ROAR = "lions roar"
    LOTUS_FLUTE = "lotus flute"
    MEGAPHONE = "megaphone"
    POLICE_WHISTLE = "police whistle"
    SIREN = "siren"
    SLIDE_WHISTLE = "slide whistle"
    THUNDER_SHEET = "thunder sheet"
    WIND_MACHINE = "wind machine"
    WIND_WHISTLE = "wind whistle"


@dataclass
class Empty:
    """
    The empty type represents an empty element with no attributes.
    """
    class Meta:
        name = "empty"


class EnclosureShape(Enum):
    """The enclosure-shape type describes the shape and presence / absence of
    an enclosure around text or symbols.

    A bracket enclosure is similar to a rectangle with the bottom line
    missing, as is common in jazz notation. An inverted-bracket
    enclosure is similar to a rectangle with the top line missing.
    """
    RECTANGLE = "rectangle"
    SQUARE = "square"
    OVAL = "oval"
    CIRCLE = "circle"
    BRACKET = "bracket"
    INVERTED_BRACKET = "inverted-bracket"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    PENTAGON = "pentagon"
    HEXAGON = "hexagon"
    HEPTAGON = "heptagon"
    OCTAGON = "octagon"
    NONAGON = "nonagon"
    DECAGON = "decagon"
    NONE = "none"


class Fan(Enum):
    """
    The fan type represents the type of beam fanning present on a note, used to
    represent accelerandos and ritardandos.
    """
    ACCEL = "accel"
    RIT = "rit"
    NONE = "none"


@dataclass
class Feature:
    """The feature type is a part of the grouping element used for musical
    analysis.

    The type attribute represents the type of the feature and the
    element content represents its value. This type is flexible to allow
    for different analyses.
    """
    class Meta:
        name = "feature"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


class FermataShape(Enum):
    """The fermata-shape type represents the shape of the fermata sign.

    The empty value is equivalent to the normal value.
    """
    NORMAL = "normal"
    ANGLED = "angled"
    SQUARE = "square"
    DOUBLE_ANGLED = "double-angled"
    DOUBLE_SQUARE = "double-square"
    DOUBLE_DOT = "double-dot"
    HALF_CURVE = "half-curve"
    CURLEW = "curlew"
    VALUE = ""


class FontStyle(Enum):
    """
    The font-style type represents a simplified version of the CSS font-style
    property.
    """
    NORMAL = "normal"
    ITALIC = "italic"


class FontWeight(Enum):
    """
    The font-weight type represents a simplified version of the CSS font-weight
    property.
    """
    NORMAL = "normal"
    BOLD = "bold"


class GlassValue(Enum):
    """
    The glass-value type represents pictograms for glass percussion
    instruments.
    """
    GLASS_HARMONICA = "glass harmonica"
    GLASS_HARP = "glass harp"
    WIND_CHIMES = "wind chimes"


@dataclass
class Glyph:
    """The glyph element represents what SMuFL glyph should be used for
    different variations of symbols that are semantically identical.

    The type attribute specifies what type of glyph is being defined.
    The element value specifies what SMuFL glyph to use, including
    recommended stylistic alternates. The SMuFL glyph name should match
    the type. For instance, a type of quarter-rest would use values
    restQuarter, restQuarterOld, or restQuarterZ. A type of g-clef-
    ottava-bassa would use values gClef8vb, gClef8vbOld, or
    gClef8vbCClef. A type of octave-shift-up-8 would use values ottava,
    ottavaBassa, ottavaBassaBa, ottavaBassaVb, or octaveBassa.
    """
    class Meta:
        name = "glyph"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class GroupBarlineValue(Enum):
    """
    The group-barline-value type indicates if the group should have common
    barlines.
    """
    YES = "yes"
    NO = "no"
    MENSURSTRICH = "Mensurstrich"


class GroupSymbolValue(Enum):
    """
    The group-symbol-value type indicates how the symbol for a group or multi-
    staff part is indicated in the score.
    """
    NONE = "none"
    BRACE = "brace"
    LINE = "line"
    BRACKET = "bracket"
    SQUARE = "square"


class HandbellValue(Enum):
    """
    The handbell-value type represents the type of handbell technique being
    notated.
    """
    BELLTREE = "belltree"
    DAMP = "damp"
    ECHO = "echo"
    GYRO = "gyro"
    HAND_MARTELLATO = "hand martellato"
    MALLET_LIFT = "mallet lift"
    MALLET_TABLE = "mallet table"
    MARTELLATO = "martellato"
    MARTELLATO_LIFT = "martellato lift"
    MUTED_MARTELLATO = "muted martellato"
    PLUCK_LIFT = "pluck lift"
    SWING = "swing"


class HarmonClosedLocation(Enum):
    """
    The harmon-closed-location type indicates which portion of the symbol is
    filled in when the corresponding harmon-closed-value is half.
    """
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"
    TOP = "top"


class HarmonClosedValue(Enum):
    """
    The harmon-closed-value type represents whether the harmon mute is closed,
    open, or half-open.
    """
    YES = "yes"
    NO = "no"
    HALF = "half"


class HarmonyArrangement(Enum):
    """The harmony-arrangement type indicates how stacked chords and bass notes
    are displayed within a harmony element.

    The vertical value specifies that the second element appears below
    the first. The horizontal value specifies that the second element
    appears to the right of the first. The diagonal value specifies that
    the second element appears both below and to the right of the first.
    """
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"


class HarmonyType(Enum):
    """The harmony-type type differentiates different types of harmonies when
    alternate harmonies are possible.

    Explicit harmonies have all note present in the music; implied have
    some notes missing but implied; alternate represents alternate
    analyses.
    """
    EXPLICIT = "explicit"
    IMPLIED = "implied"
    ALTERNATE = "alternate"


class HoleClosedLocation(Enum):
    """
    The hole-closed-location type indicates which portion of the hole is filled
    in when the corresponding hole-closed-value is half.
    """
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"
    TOP = "top"


class HoleClosedValue(Enum):
    """
    The hole-closed-value type represents whether the hole is closed, open, or
    half-open.
    """
    YES = "yes"
    NO = "no"
    HALF = "half"


@dataclass
class Instrument:
    """The instrument type distinguishes between score-instrument elements in a
    score-part.

    The id attribute is an IDREF back to the score-instrument ID. If
    multiple score-instruments are specified in a score-part, there
    should be an instrument element for each note in the part. Notes
    that are shared between multiple score-instruments can have more
    than one instrument element.
    """
    class Meta:
        name = "instrument"

    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class InstrumentLink:
    """Multiple part-link elements can link a condensed part within a score
    file to multiple MusicXML parts files.

    For example, a "Clarinet 1 and 2" part in a score file could link to
    separate "Clarinet 1" and "Clarinet 2" part files. The instrument-
    link type distinguish which of the score-instruments within a score-
    part are in which part file. The instrument-link id attribute refers
    to a score-instrument id attribute.
    """
    class Meta:
        name = "instrument-link"

    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class KindValue(Enum):
    """A kind-value indicates the type of chord.

    Degree elements can then add, subtract, or alter from these starting points. Values include:
    Triads:
    major (major third, perfect fifth)
    minor (minor third, perfect fifth)
    augmented (major third, augmented fifth)
    diminished (minor third, diminished fifth)
    Sevenths:
    dominant (major triad, minor seventh)
    major-seventh (major triad, major seventh)
    minor-seventh (minor triad, minor seventh)
    diminished-seventh (diminished triad, diminished seventh)
    augmented-seventh (augmented triad, minor seventh)
    half-diminished (diminished triad, minor seventh)
    major-minor (minor triad, major seventh)
    Sixths:
    major-sixth (major triad, added sixth)
    minor-sixth (minor triad, added sixth)
    Ninths:
    dominant-ninth (dominant-seventh, major ninth)
    major-ninth (major-seventh, major ninth)
    minor-ninth (minor-seventh, major ninth)
    11ths (usually as the basis for alteration):
    dominant-11th (dominant-ninth, perfect 11th)
    major-11th (major-ninth, perfect 11th)
    minor-11th (minor-ninth, perfect 11th)
    13ths (usually as the basis for alteration):
    dominant-13th (dominant-11th, major 13th)
    major-13th (major-11th, major 13th)
    minor-13th (minor-11th, major 13th)
    Suspended:
    suspended-second (major second, perfect fifth)
    suspended-fourth (perfect fourth, perfect fifth)
    Functional sixths:
    Neapolitan
    Italian
    French
    German
    Other:
    pedal (pedal-point bass)
    power (perfect fifth)
    Tristan
    The "other" kind is used when the harmony is entirely composed of add elements.
    The "none" kind is used to explicitly encode absence of chords or functional harmony. In this case, the root, numeral, or function element has no meaning. When using the root or numeral element, the root-step or numeral-step text attribute should be set to the empty string to keep the root or numeral from being displayed.
    """
    MAJOR = "major"
    MINOR = "minor"
    AUGMENTED = "augmented"
    DIMINISHED = "diminished"
    DOMINANT = "dominant"
    MAJOR_SEVENTH = "major-seventh"
    MINOR_SEVENTH = "minor-seventh"
    DIMINISHED_SEVENTH = "diminished-seventh"
    AUGMENTED_SEVENTH = "augmented-seventh"
    HALF_DIMINISHED = "half-diminished"
    MAJOR_MINOR = "major-minor"
    MAJOR_SIXTH = "major-sixth"
    MINOR_SIXTH = "minor-sixth"
    DOMINANT_NINTH = "dominant-ninth"
    MAJOR_NINTH = "major-ninth"
    MINOR_NINTH = "minor-ninth"
    DOMINANT_11TH = "dominant-11th"
    MAJOR_11TH = "major-11th"
    MINOR_11TH = "minor-11th"
    DOMINANT_13TH = "dominant-13th"
    MAJOR_13TH = "major-13th"
    MINOR_13TH = "minor-13th"
    SUSPENDED_SECOND = "suspended-second"
    SUSPENDED_FOURTH = "suspended-fourth"
    NEAPOLITAN = "Neapolitan"
    ITALIAN = "Italian"
    FRENCH = "French"
    GERMAN = "German"
    PEDAL = "pedal"
    POWER = "power"
    TRISTAN = "Tristan"
    OTHER = "other"
    NONE = "none"


class LeftCenterRight(Enum):
    """
    The left-center-right type is used to define horizontal alignment and text
    justification.
    """
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class LeftRight(Enum):
    """
    The left-right type is used to indicate whether one element appears to the
    left or the right of another element.
    """
    LEFT = "left"
    RIGHT = "right"


class LineEnd(Enum):
    """
    The line-end type specifies if there is a jog up or down (or both), an
    arrow, or nothing at the start or end of a bracket.
    """
    UP = "up"
    DOWN = "down"
    BOTH = "both"
    ARROW = "arrow"
    NONE = "none"


class LineLength(Enum):
    """
    The line-length type distinguishes between different line lengths for doit,
    falloff, plop, and scoop articulations.
    """
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class LineShape(Enum):
    """
    The line-shape type distinguishes between straight and curved lines.
    """
    STRAIGHT = "straight"
    CURVED = "curved"


class LineType(Enum):
    """
    The line-type type distinguishes between solid, dashed, dotted, and wavy
    lines.
    """
    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    WAVY = "wavy"


@dataclass
class LineWidth:
    """The line-width type indicates the width of a line type in tenths.

    The type attribute defines what type of line is being defined.
    Values include beam, bracket, dashes, enclosure, ending, extend,
    heavy barline, leger, light barline, octave shift, pedal, slur
    middle, slur tip, staff, stem, tie middle, tie tip, tuplet bracket,
    and wedge. The text content is expressed in tenths.
    """
    class Meta:
        name = "line-width"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class MarginType(Enum):
    """
    The margin-type type specifies whether margins apply to even page, odd
    pages, or both.
    """
    ODD = "odd"
    EVEN = "even"
    BOTH = "both"


@dataclass
class MeasureLayout:
    """The measure-layout type includes the horizontal distance from the
    previous measure.

    It applies to the current measure only.

    :ivar measure_distance: The measure-distance element specifies the
        horizontal distance from the previous measure. This value is
        only used for systems where there is horizontal whitespace in
        the middle of a system, as in systems with codas. To specify the
        measure width, use the width attribute of the measure element.
    """
    class Meta:
        name = "measure-layout"

    measure_distance: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "measure-distance",
            "type": "Element",
        }
    )


class MeasureNumberingValue(Enum):
    """
    The measure-numbering-value type describes how measure numbers are
    displayed on this part: no numbers, numbers every measure, or numbers every
    system.
    """
    NONE = "none"
    MEASURE = "measure"
    SYSTEM = "system"


class MembraneValue(Enum):
    """
    The membrane-value type represents pictograms for membrane percussion
    instruments.
    """
    BASS_DRUM = "bass drum"
    BASS_DRUM_ON_SIDE = "bass drum on side"
    BONGOS = "bongos"
    CHINESE_TOMTOM = "Chinese tomtom"
    CONGA_DRUM = "conga drum"
    CUICA = "cuica"
    GOBLET_DRUM = "goblet drum"
    INDO_AMERICAN_TOMTOM = "Indo-American tomtom"
    JAPANESE_TOMTOM = "Japanese tomtom"
    MILITARY_DRUM = "military drum"
    SNARE_DRUM = "snare drum"
    SNARE_DRUM_SNARES_OFF = "snare drum snares off"
    TABLA = "tabla"
    TAMBOURINE = "tambourine"
    TENOR_DRUM = "tenor drum"
    TIMBALES = "timbales"
    TOMTOM = "tomtom"


class MetalValue(Enum):
    """The metal-value type represents pictograms for metal percussion
    instruments.

    The hi-hat value refers to a pictogram like Stone's high-hat cymbals
    but without the long vertical line at the bottom.
    """
    AGOGO = "agogo"
    ALMGLOCKEN = "almglocken"
    BELL = "bell"
    BELL_PLATE = "bell plate"
    BELL_TREE = "bell tree"
    BRAKE_DRUM = "brake drum"
    CENCERRO = "cencerro"
    CHAIN_RATTLE = "chain rattle"
    CHINESE_CYMBAL = "Chinese cymbal"
    COWBELL = "cowbell"
    CRASH_CYMBALS = "crash cymbals"
    CROTALE = "crotale"
    CYMBAL_TONGS = "cymbal tongs"
    DOMED_GONG = "domed gong"
    FINGER_CYMBALS = "finger cymbals"
    FLEXATONE = "flexatone"
    GONG = "gong"
    HI_HAT = "hi-hat"
    HIGH_HAT_CYMBALS = "high-hat cymbals"
    HANDBELL = "handbell"
    JAW_HARP = "jaw harp"
    JINGLE_BELLS = "jingle bells"
    MUSICAL_SAW = "musical saw"
    SHELL_BELLS = "shell bells"
    SISTRUM = "sistrum"
    SIZZLE_CYMBAL = "sizzle cymbal"
    SLEIGH_BELLS = "sleigh bells"
    SUSPENDED_CYMBAL = "suspended cymbal"
    TAM_TAM = "tam tam"
    TAM_TAM_WITH_BEATER = "tam tam with beater"
    TRIANGLE = "triangle"
    VIETNAMESE_HAT = "Vietnamese hat"


@dataclass
class MidiDevice:
    """The midi-device type corresponds to the DeviceName meta event in
    Standard MIDI Files.

    The optional port attribute is a number from 1 to 16 that can be
    used with the unofficial MIDI 1.0 port (or cable) meta event. Unlike
    the DeviceName meta event, there can be multiple midi-device
    elements per MusicXML part. The optional id attribute refers to the
    score-instrument assigned to this device. If missing, the device
    assignment affects all score-instrument elements in the score-part.
    """
    class Meta:
        name = "midi-device"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    port: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class MidiInstrument:
    """The midi-instrument type defines MIDI 1.0 instrument playback.

    The midi-instrument element can be a part of either the score-
    instrument element at the start of a part, or the sound element
    within a part. The id attribute refers to the score-instrument
    affected by the change.

    :ivar midi_channel: The midi-channel element specifies a MIDI 1.0
        channel numbers ranging from 1 to 16.
    :ivar midi_name: The midi-name element corresponds to a ProgramName
        meta-event within a Standard MIDI File.
    :ivar midi_bank: The midi-bank element specifies a MIDI 1.0 bank
        number ranging from 1 to 16,384.
    :ivar midi_program: The midi-program element specifies a MIDI 1.0
        program number ranging from 1 to 128.
    :ivar midi_unpitched: For unpitched instruments, the midi-unpitched
        element specifies a MIDI 1.0 note number ranging from 1 to 128.
        It is usually used with MIDI banks for percussion. Note that
        MIDI 1.0 note numbers are generally specified from 0 to 127
        rather than the 1 to 128 numbering used in this element.
    :ivar volume: The volume element value is a percentage of the
        maximum ranging from 0 to 100, with decimal values allowed. This
        corresponds to a scaling value for the MIDI 1.0 channel volume
        controller.
    :ivar pan: The pan and elevation elements allow placing of sound in
        a 3-D space relative to the listener. Both are expressed in
        degrees ranging from -180 to 180. For pan, 0 is straight ahead,
        -90 is hard left, 90 is hard right, and -180 and 180 are
        directly behind the listener.
    :ivar elevation: The elevation and pan elements allow placing of
        sound in a 3-D space relative to the listener. Both are
        expressed in degrees ranging from -180 to 180. For elevation, 0
        is level with the listener, 90 is directly above, and -90 is
        directly below.
    :ivar id:
    """
    class Meta:
        name = "midi-instrument"

    midi_channel: Optional[int] = field(
        default=None,
        metadata={
            "name": "midi-channel",
            "type": "Element",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    midi_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "midi-name",
            "type": "Element",
        }
    )
    midi_bank: Optional[int] = field(
        default=None,
        metadata={
            "name": "midi-bank",
            "type": "Element",
            "min_inclusive": 1,
            "max_inclusive": 16384,
        }
    )
    midi_program: Optional[int] = field(
        default=None,
        metadata={
            "name": "midi-program",
            "type": "Element",
            "min_inclusive": 1,
            "max_inclusive": 128,
        }
    )
    midi_unpitched: Optional[int] = field(
        default=None,
        metadata={
            "name": "midi-unpitched",
            "type": "Element",
            "min_inclusive": 1,
            "max_inclusive": 128,
        }
    )
    volume: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    pan: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    elevation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class MiscellaneousField:
    """If a program has other metadata not yet supported in the MusicXML
    format, each type of metadata can go in a miscellaneous-field element.

    The required name attribute indicates the type of metadata the
    element content represents.
    """
    class Meta:
        name = "miscellaneous-field"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class Mute(Enum):
    """The mute type represents muting for different instruments, including
    brass, winds, and strings.

    The on and off values are used for undifferentiated mutes. The
    remaining values represent specific mutes.
    """
    ON = "on"
    OFF = "off"
    STRAIGHT = "straight"
    CUP = "cup"
    HARMON_NO_STEM = "harmon-no-stem"
    HARMON_STEM = "harmon-stem"
    BUCKET = "bucket"
    PLUNGER = "plunger"
    HAT = "hat"
    SOLOTONE = "solotone"
    PRACTICE = "practice"
    STOP_MUTE = "stop-mute"
    STOP_HAND = "stop-hand"
    ECHO = "echo"
    PALM = "palm"


class NoteSizeType(Enum):
    """The note-size-type type indicates the type of note being defined by a
    note-size element.

    The grace-cue type is used for notes of grace-cue size. The grace
    type is used for notes of cue size that include a grace element. The
    cue type is used for all other notes with cue size, whether defined
    explicitly or implicitly via a cue element. The large type is used
    for notes of large size.
    """
    CUE = "cue"
    GRACE = "grace"
    GRACE_CUE = "grace-cue"
    LARGE = "large"


class NoteTypeValue(Enum):
    """
    The note-type-value type is used for the MusicXML type element and
    represents the graphic note type, from 1024th (shortest) to maxima
    (longest).
    """
    VALUE_1024TH = "1024th"
    VALUE_512TH = "512th"
    VALUE_256TH = "256th"
    VALUE_128TH = "128th"
    VALUE_64TH = "64th"
    VALUE_32ND = "32nd"
    VALUE_16TH = "16th"
    EIGHTH = "eighth"
    QUARTER = "quarter"
    HALF = "half"
    WHOLE = "whole"
    BREVE = "breve"
    LONG = "long"
    MAXIMA = "maxima"


class NoteheadValue(Enum):
    """The notehead-value type indicates shapes other than the open and closed
    ovals associated with note durations.

    The values do, re, mi, fa, fa up, so, la, and ti correspond to
    Aikin's 7-shape system.  The fa up shape is typically used with
    upstems; the fa shape is typically used with downstems or no stems.
    The arrow shapes differ from triangle and inverted triangle by being
    centered on the stem. Slashed and back slashed notes include both
    the normal notehead and a slash. The triangle shape has the tip of
    the triangle pointing up; the inverted triangle shape has the tip of
    the triangle pointing down. The left triangle shape is a right
    triangle with the hypotenuse facing up and to the left. The other
    notehead covers noteheads other than those listed here. It is
    usually used in combination with the smufl attribute to specify a
    particular SMuFL notehead. The smufl attribute may be used with any
    notehead value to help specify the appearance of symbols that share
    the same MusicXML semantics. Noteheads in the SMuFL Note name
    noteheads and Note name noteheads supplement ranges (U+E150–U+E1AF
    and U+EEE0–U+EEFF) should not use the smufl attribute or the "other"
    value, but instead use the notehead-text element.
    """
    SLASH = "slash"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"
    SQUARE = "square"
    CROSS = "cross"
    X = "x"
    CIRCLE_X = "circle-x"
    INVERTED_TRIANGLE = "inverted triangle"
    ARROW_DOWN = "arrow down"
    ARROW_UP = "arrow up"
    CIRCLED = "circled"
    SLASHED = "slashed"
    BACK_SLASHED = "back slashed"
    NORMAL = "normal"
    CLUSTER = "cluster"
    CIRCLE_DOT = "circle dot"
    LEFT_TRIANGLE = "left triangle"
    RECTANGLE = "rectangle"
    NONE = "none"
    DO = "do"
    RE = "re"
    MI = "mi"
    FA = "fa"
    FA_UP = "fa up"
    SO = "so"
    LA = "la"
    TI = "ti"
    OTHER = "other"


class NumberOrNormalValue(Enum):
    NORMAL = "normal"


class NumeralMode(Enum):
    """The numeral-mode type specifies the mode similar to the mode type, but
    with a restricted set of values.

    The different minor values are used to interpret numeral-root values
    of 6 and 7 when present in a minor key. The harmonic minor value
    sharpens the 7 and the melodic minor value sharpens both 6 and 7. If
    a minor mode is used without qualification, either in the mode or
    numeral-mode elements, natural minor is used.
    """
    MAJOR = "major"
    MINOR = "minor"
    NATURAL_MINOR = "natural minor"
    MELODIC_MINOR = "melodic minor"
    HARMONIC_MINOR = "harmonic minor"


class OnOff(Enum):
    """
    The on-off type is used for notation elements such as string mutes.
    """
    ON = "on"
    OFF = "off"


@dataclass
class OtherAppearance:
    """The other-appearance type is used to define any graphical settings not
    yet in the current version of the MusicXML format.

    This allows extended representation, though without application
    interoperability.
    """
    class Meta:
        name = "other-appearance"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class OtherListening:
    """The other-listening type represents other types of listening control and
    interaction.

    The required type attribute indicates the type of listening to which
    the element content applies. The optional player and time-only
    attributes restrict the element to apply to a single player or set
    of times through a repeated section, respectively.
    """
    class Meta:
        name = "other-listening"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    player: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )


@dataclass
class OtherPlay:
    """The other-play element represents other types of playback.

    The required type attribute indicates the type of playback to which
    the element content applies.
    """
    class Meta:
        name = "other-play"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class OtherText:
    """The other-text type represents a text element with a smufl attribute
    group.

    This type is used by MusicXML direction extension elements to allow
    specification of specific SMuFL glyphs without needed to add every
    glyph as a MusicXML element.
    """
    class Meta:
        name = "other-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


class OverUnder(Enum):
    """
    The over-under type is used to indicate whether the tips of curved lines
    such as slurs and ties are overhand (tips down) or underhand (tips up).
    """
    OVER = "over"
    UNDER = "under"


class PedalType(Enum):
    """The pedal-type simple type is used to distinguish types of pedal
    directions.

    The start value indicates the start of a damper pedal, while the
    sostenuto value indicates the start of a sostenuto pedal. The other
    values can be used with either the damper or sostenuto pedal. The
    soft pedal is not included here because there is no special symbol
    or graphic used for it beyond what can be specified with words and
    bracket elements. The change, continue, discontinue, and resume
    types are used when the line attribute is yes. The change type
    indicates a pedal lift and retake indicated with an inverted V
    marking. The continue type allows more precise formatting across
    system breaks and for more complex pedaling lines. The discontinue
    type indicates the end of a pedal line that does not include the
    explicit lift represented by the stop type. The resume type
    indicates the start of a pedal line that does not include the
    downstroke represented by the start type. It can be used when a line
    resumes after being discontinued, or to start a pedal line that is
    preceded by a text or symbol representation of the pedal.
    """
    START = "start"
    STOP = "stop"
    SOSTENUTO = "sostenuto"
    CHANGE = "change"
    CONTINUE = "continue"
    DISCONTINUE = "discontinue"
    RESUME = "resume"


class PitchedValue(Enum):
    """The pitched-value type represents pictograms for pitched percussion
    instruments.

    The chimes and tubular chimes values distinguish the single-line and
    double-line versions of the pictogram.
    """
    CELESTA = "celesta"
    CHIMES = "chimes"
    GLOCKENSPIEL = "glockenspiel"
    LITHOPHONE = "lithophone"
    MALLET = "mallet"
    MARIMBA = "marimba"
    STEEL_DRUMS = "steel drums"
    TUBAPHONE = "tubaphone"
    TUBULAR_CHIMES = "tubular chimes"
    VIBRAPHONE = "vibraphone"
    XYLOPHONE = "xylophone"


@dataclass
class Player:
    """The player type allows for multiple players per score-part for use in
    listening applications.

    One player may play multiple instruments, while a single instrument
    may include multiple players in divisi sections.

    :ivar player_name: The player-name element is typically used within
        a software application, rather than appearing on the printed
        page of a score.
    :ivar id:
    """
    class Meta:
        name = "player"

    player_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "player-name",
            "type": "Element",
            "required": True,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


class PositiveIntegerOrEmptyValue(Enum):
    VALUE = ""


class PrincipalVoiceSymbol(Enum):
    """The principal-voice-symbol type represents the type of symbol used to
    indicate a principal or secondary voice.

    The "plain" value represents a plain square bracket. The value of
    "none" is used for analysis markup when the principal-voice element
    does not have a corresponding appearance in the score.
    """
    HAUPTSTIMME = "Hauptstimme"
    NEBENSTIMME = "Nebenstimme"
    PLAIN = "plain"
    NONE = "none"


class RightLeftMiddle(Enum):
    """
    The right-left-middle type is used to specify barline location.
    """
    RIGHT = "right"
    LEFT = "left"
    MIDDLE = "middle"


@dataclass
class Scaling:
    """Margins, page sizes, and distances are all measured in tenths to keep
    MusicXML data in a consistent coordinate system as much as possible.

    The translation to absolute units is done with the scaling type,
    which specifies how many millimeters are equal to how many tenths.
    For a staff height of 7 mm, millimeters would be set to 7 while
    tenths is set to 40. The ability to set a formula rather than a
    single scaling factor helps avoid roundoff errors.
    """
    class Meta:
        name = "scaling"

    millimeters: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    tenths: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )


class SemiPitched(Enum):
    """
    The semi-pitched type represents categories of indefinite pitch for
    percussion instruments.
    """
    HIGH = "high"
    MEDIUM_HIGH = "medium-high"
    MEDIUM = "medium"
    MEDIUM_LOW = "medium-low"
    LOW = "low"
    VERY_LOW = "very-low"


class ShowFrets(Enum):
    """The show-frets type indicates whether to show tablature frets as numbers
    (0, 1, 2) or letters (a, b, c).

    The default choice is numbers.
    """
    NUMBERS = "numbers"
    LETTERS = "letters"


class ShowTuplet(Enum):
    """
    The show-tuplet type indicates whether to show a part of a tuplet relating
    to the tuplet-actual element, both the tuplet-actual and tuplet-normal
    elements, or neither.
    """
    ACTUAL = "actual"
    BOTH = "both"
    NONE = "none"


class StaffDivideSymbol(Enum):
    """The staff-divide-symbol type is used for staff division symbols.

    The down, up, and up-down values correspond to SMuFL code points
    U+E00B, U+E00C, and U+E00D respectively.
    """
    DOWN = "down"
    UP = "up"
    UP_DOWN = "up-down"


@dataclass
class StaffLayout:
    """Staff layout includes the vertical distance from the bottom line of the
    previous staff in this system to the top line of the staff specified by the
    number attribute.

    The optional number attribute refers to staff numbers within the
    part, from top to bottom on the system. A value of 1 is used if not
    present. When used in the defaults element, the values apply to all
    systems in all parts. When used in the print element, the values
    apply to the current system only. This value is ignored for the
    first staff in a system.
    """
    class Meta:
        name = "staff-layout"

    staff_distance: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "staff-distance",
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StaffSize:
    """The staff-size element indicates how large a staff space is on this
    staff, expressed as a percentage of the work's default scaling.

    Values less than 100 make the staff space smaller while values over
    100 make the staff space larger. A staff-type of cue, ossia, or
    editorial implies a staff-size of less than 100, but the exact value
    is implementation-dependent unless specified here. Staff size
    affects staff height only, not the relationship of the staff to the
    left and right margins. In some cases, a staff-size different than
    100 also scales the notation on the staff, such as with a cue staff.
    In other cases, such as percussion staves, the lines may be more
    widely spaced without scaling the notation on the staff. The scaling
    attribute allows these two cases to be distinguished. It specifies
    the percentage scaling that applies to the notation. Values less
    that 100 make the notation smaller while values over 100 make the
    notation larger. The staff-size content and scaling attribute are
    both non-negative decimal values.
    """
    class Meta:
        name = "staff-size"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
            "min_inclusive": Decimal("0"),
        }
    )
    scaling: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
        }
    )


class StaffType(Enum):
    """The staff-type value can be ossia, editorial, cue, alternate, or
    regular.

    An ossia staff represents music that can be played instead of what
    appears on the regular staff. An editorial staff also represents
    musical alternatives, but is created by an editor rather than the
    composer. It can be used for suggested interpretations or
    alternatives from other sources. A cue staff represents music from
    another part. An alternate staff shares the same music as the prior
    staff, but displayed differently (e.g., treble and bass clef,
    standard notation and tablature). It is not included in playback. An
    alternate staff provides more information to an application reading
    a file than encoding the same music in separate parts, so its use is
    preferred in this situation if feasible. A regular staff is the
    standard default staff-type.
    """
    OSSIA = "ossia"
    EDITORIAL = "editorial"
    CUE = "cue"
    ALTERNATE = "alternate"
    REGULAR = "regular"


class StartNote(Enum):
    """
    The start-note type describes the starting note of trills and mordents for
    playback, relative to the current note.
    """
    UPPER = "upper"
    MAIN = "main"
    BELOW = "below"


class StartStop(Enum):
    """The start-stop type is used for an attribute of musical elements that
    can either start or stop, such as tuplets.

    The values of start and stop refer to how an element appears in
    musical score order, not in MusicXML document order. An element with
    a stop attribute may precede the corresponding element with a start
    attribute within a MusicXML document. This is particularly common in
    multi-staff music. For example, the stopping point for a tuplet may
    appear in staff 1 before the starting point for the tuplet appears
    in staff 2 later in the document. When multiple elements with the
    same tag are used within the same note, their order within the
    MusicXML document should match the musical score order.
    """
    START = "start"
    STOP = "stop"


class StartStopChangeContinue(Enum):
    """
    The start-stop-change-continue type is used to distinguish types of pedal
    directions.
    """
    START = "start"
    STOP = "stop"
    CHANGE = "change"
    CONTINUE = "continue"


class StartStopContinue(Enum):
    """The start-stop-continue type is used for an attribute of musical
    elements that can either start or stop, but also need to refer to an
    intermediate point in the symbol, as for complex slurs or for formatting of
    symbols across system breaks.

    The values of start, stop, and continue refer to how an element
    appears in musical score order, not in MusicXML document order. An
    element with a stop attribute may precede the corresponding element
    with a start attribute within a MusicXML document. This is
    particularly common in multi-staff music. For example, the stopping
    point for a slur may appear in staff 1 before the starting point for
    the slur appears in staff 2 later in the document. When multiple
    elements with the same tag are used within the same note, their
    order within the MusicXML document should match the musical score
    order. For example, a note that marks both the end of one slur and
    the start of a new slur should have the incoming slur element with a
    type of stop precede the outgoing slur element with a type of start.
    """
    START = "start"
    STOP = "stop"
    CONTINUE = "continue"


class StartStopDiscontinue(Enum):
    """The start-stop-discontinue type is used to specify ending types.

    Typically, the start type is associated with the left barline of the
    first measure in an ending. The stop and discontinue types are
    associated with the right barline of the last measure in an ending.
    Stop is used when the ending mark concludes with a downward jog, as
    is typical for first endings. Discontinue is used when there is no
    downward jog, as is typical for second endings that do not conclude
    a piece.
    """
    START = "start"
    STOP = "stop"
    DISCONTINUE = "discontinue"


class StartStopSingle(Enum):
    """The start-stop-single type is used for an attribute of musical elements
    that can be used for either multi-note or single-note musical elements, as
    for groupings.

    When multiple elements with the same tag are used within the same
    note, their order within the MusicXML document should match the
    musical score order.
    """
    START = "start"
    STOP = "stop"
    SINGLE = "single"


class StemValue(Enum):
    """
    The stem-value type represents the notated stem direction.
    """
    DOWN = "down"
    UP = "up"
    DOUBLE = "double"
    NONE = "none"


class Step(Enum):
    """
    The step type represents a step of the diatonic scale, represented using
    the English letters A through G.
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"


class StickLocation(Enum):
    """
    The stick-location type represents pictograms for the location of sticks,
    beaters, or mallets on cymbals, gongs, drums, and other instruments.
    """
    CENTER = "center"
    RIM = "rim"
    CYMBAL_BELL = "cymbal bell"
    CYMBAL_EDGE = "cymbal edge"


class StickMaterial(Enum):
    """
    The stick-material type represents the material being displayed in a stick
    pictogram.
    """
    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    SHADED = "shaded"
    X = "x"


class StickType(Enum):
    """
    The stick-type type represents the shape of pictograms where the material
    in the stick, mallet, or beater is represented in the pictogram.
    """
    BASS_DRUM = "bass drum"
    DOUBLE_BASS_DRUM = "double bass drum"
    GLOCKENSPIEL = "glockenspiel"
    GUM = "gum"
    HAMMER = "hammer"
    SUPERBALL = "superball"
    TIMPANI = "timpani"
    WOUND = "wound"
    XYLOPHONE = "xylophone"
    YARN = "yarn"


class SwingTypeValue(Enum):
    """
    The swing-type-value type specifies the note type, either eighth or 16th,
    to which the ratio defined in the swing element is applied.
    """
    VALUE_16TH = "16th"
    EIGHTH = "eighth"


class Syllabic(Enum):
    """Lyric hyphenation is indicated by the syllabic type.

    The single, begin, end, and middle values represent single-syllable
    words, word-beginning syllables, word-ending syllables, and mid-word
    syllables, respectively.
    """
    SINGLE = "single"
    BEGIN = "begin"
    END = "end"
    MIDDLE = "middle"


class SymbolSize(Enum):
    """
    The symbol-size type is used to distinguish between full, cue sized, grace
    cue sized, and oversized symbols.
    """
    FULL = "full"
    CUE = "cue"
    GRACE_CUE = "grace-cue"
    LARGE = "large"


class SyncType(Enum):
    """The sync-type type specifies the style that a score following
    application should use to synchronize an accompaniment with a performer.

    The none type indicates no synchronization to the performer. The
    tempo type indicates synchronization based on the performer tempo
    rather than individual events in the score. The event type indicates
    synchronization by following the performance of individual events in
    the score rather than the performer tempo. The mostly-tempo and
    mostly-event types combine these two approaches, with mostly-tempo
    giving more weight to tempo and mostly-event giving more weight to
    performed events. The always-event type provides the strictest
    synchronization by not being forgiving of missing performed events.
    """
    NONE = "none"
    TEMPO = "tempo"
    MOSTLY_TEMPO = "mostly-tempo"
    MOSTLY_EVENT = "mostly-event"
    EVENT = "event"
    ALWAYS_EVENT = "always-event"


@dataclass
class SystemMargins:
    """System margins are relative to the page margins.

    Positive values indent and negative values reduce the margin size.
    """
    class Meta:
        name = "system-margins"

    left_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "left-margin",
            "type": "Element",
            "required": True,
        }
    )
    right_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "right-margin",
            "type": "Element",
            "required": True,
        }
    )


class SystemRelation(Enum):
    """The system-relation type distinguishes elements that are associated with
    a system rather than the particular part where the element appears.

    A value of only-top indicates that the element should appear only on
    the top part of the current system. A value of also-top indicates
    that the element should appear on both the current part and the top
    part of the current system. If this value appears in a score, when
    parts are created the element should only appear once in this part,
    not twice. A value of none indicates that the element is associated
    only with the current part, not with the system.
    """
    ONLY_TOP = "only-top"
    ALSO_TOP = "also-top"
    NONE = "none"


class SystemRelationNumber(Enum):
    """The system-relation-number type distinguishes measure numbers that are
    associated with a system rather than the particular part where the element
    appears.

    A value of only-top or only-bottom indicates that the number should
    appear only on the top or bottom part of the current system,
    respectively. A value of also-top or also-bottom indicates that the
    number should appear on both the current part and the top or bottom
    part of the current system, respectively. If these values appear in
    a score, when parts are created the number should only appear once
    in this part, not twice. A value of none indicates that the number
    is associated only with the current part, not with the system.
    """
    ONLY_TOP = "only-top"
    ONLY_BOTTOM = "only-bottom"
    ALSO_TOP = "also-top"
    ALSO_BOTTOM = "also-bottom"
    NONE = "none"


class TapHand(Enum):
    """The tap-hand type represents the symbol to use for a tap element.

    The left and right values refer to the SMuFL guitarLeftHandTapping
    and guitarRightHandTapping glyphs respectively.
    """
    LEFT = "left"
    RIGHT = "right"


class TextDirection(Enum):
    """The text-direction type is used to adjust and override the Unicode
    bidirectional text algorithm, similar to the Directionality data category
    in the W3C Internationalization Tag Set recommendation.

    Values are ltr (left-to-right embed), rtl (right-to-left embed), lro
    (left-to-right bidi-override), and rlo (right-to-left bidi-
    override). The default value is ltr. This type is typically used by
    applications that store text in left-to-right visual order rather
    than logical order. Such applications can use the lro value to
    better communicate with other applications that more fully support
    bidirectional text.
    """
    LTR = "ltr"
    RTL = "rtl"
    LRO = "lro"
    RLO = "rlo"


class TiedType(Enum):
    """The tied-type type is used as an attribute of the tied element to
    specify where the visual representation of a tie begins and ends.

    A tied element which joins two notes of the same pitch can be
    specified with tied-type start on the first note and tied-type stop
    on the second note. To indicate a note should be undamped, use a
    single tied element with tied-type let-ring. For other ties that are
    visually attached to a single note, such as a tie leading into or
    out of a repeated section or coda, use two tied elements on the same
    note, one start and one stop. In start-stop cases, ties can add more
    elements using a continue type. This is typically used to specify
    the formatting of cross-system ties. When multiple elements with the
    same tag are used within the same note, their order within the
    MusicXML document should match the musical score order. For example,
    a note with a tie at the end of a first ending should have the tied
    element with a type of start precede the tied element with a type of
    stop.
    """
    START = "start"
    STOP = "stop"
    CONTINUE = "continue"
    LET_RING = "let-ring"


class TimeRelation(Enum):
    """
    The time-relation type indicates the symbol used to represent the
    interchangeable aspect of dual time signatures.
    """
    PARENTHESES = "parentheses"
    BRACKET = "bracket"
    EQUALS = "equals"
    SLASH = "slash"
    SPACE = "space"
    HYPHEN = "hyphen"


class TimeSeparator(Enum):
    """The time-separator type indicates how to display the arrangement between
    the beats and beat-type values in a time signature.

    The default value is none. The horizontal, diagonal, and vertical
    values represent horizontal, diagonal lower-left to upper-right, and
    vertical lines respectively. For these values, the beats and beat-
    type values are arranged on either side of the separator line. The
    none value represents no separator with the beats and beat-type
    arranged vertically. The adjacent value represents no separator with
    the beats and beat-type arranged horizontally.
    """
    NONE = "none"
    HORIZONTAL = "horizontal"
    DIAGONAL = "diagonal"
    VERTICAL = "vertical"
    ADJACENT = "adjacent"


class TimeSymbol(Enum):
    """The time-symbol type indicates how to display a time signature.

    The normal value is the usual fractional display, and is the implied
    symbol type if none is specified. Other options are the common and
    cut time symbols, as well as a single number with an implied
    denominator. The note symbol indicates that the beat-type should be
    represented with the corresponding downstem note rather than a
    number. The dotted-note symbol indicates that the beat-type should
    be represented with a dotted downstem note that corresponds to three
    times the beat-type value, and a numerator that is one third the
    beats value.
    """
    COMMON = "common"
    CUT = "cut"
    SINGLE_NUMBER = "single-number"
    NOTE = "note"
    DOTTED_NOTE = "dotted-note"
    NORMAL = "normal"


@dataclass
class Timpani:
    """The timpani type represents the timpani pictogram.

    The smufl attribute is used to distinguish different SMuFL stylistic
    alternates.
    """
    class Meta:
        name = "timpani"

    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


class TipDirection(Enum):
    """
    The tip-direction type represents the direction in which the tip of a stick
    or beater points, using Unicode arrow terminology.
    """
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NORTHWEST = "northwest"
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


class TopBottom(Enum):
    """
    The top-bottom type is used to indicate the top or bottom part of a
    vertical shape like non-arpeggiate.
    """
    TOP = "top"
    BOTTOM = "bottom"


class TremoloType(Enum):
    """
    The tremolo-type is used to distinguish double-note, single-note, and
    unmeasured tremolos.
    """
    START = "start"
    STOP = "stop"
    SINGLE = "single"
    UNMEASURED = "unmeasured"


class TrillStep(Enum):
    """
    The trill-step type describes the alternating note of trills and mordents
    for playback, relative to the current note.
    """
    WHOLE = "whole"
    HALF = "half"
    UNISON = "unison"


class TwoNoteTurn(Enum):
    """
    The two-note-turn type describes the ending notes of trills and mordents
    for playback, relative to the current note.
    """
    WHOLE = "whole"
    HALF = "half"
    NONE = "none"


@dataclass
class TypedText:
    """
    The typed-text type represents a text element with a type attribute.
    """
    class Meta:
        name = "typed-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


class UpDown(Enum):
    """
    The up-down type is used for the direction of arrows and other pointed
    symbols like vertical accents, indicating which way the tip is pointing.
    """
    UP = "up"
    DOWN = "down"


class UpDownStopContinue(Enum):
    """
    The up-down-stop-continue type is used for octave-shift elements,
    indicating the direction of the shift from their true pitched values
    because of printing difficulty.
    """
    UP = "up"
    DOWN = "down"
    STOP = "stop"
    CONTINUE = "continue"


class UprightInverted(Enum):
    """The upright-inverted type describes the appearance of a fermata element.

    The value is upright if not specified.
    """
    UPRIGHT = "upright"
    INVERTED = "inverted"


class Valign(Enum):
    """The valign type is used to indicate vertical alignment to the top,
    middle, bottom, or baseline of the text.

    If the text is on multiple lines, baseline alignment refers to the
    baseline of the lowest line of text. Defaults are implementation-
    dependent.
    """
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    BASELINE = "baseline"


class ValignImage(Enum):
    """The valign-image type is used to indicate vertical alignment for images
    and graphics, so it does not include a baseline value.

    Defaults are implementation-dependent.
    """
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"


@dataclass
class VirtualInstrument:
    """
    The virtual-instrument element defines a specific virtual instrument used
    for an instrument sound.

    :ivar virtual_library: The virtual-library element indicates the
        virtual instrument library name.
    :ivar virtual_name: The virtual-name element indicates the library-
        specific name for the virtual instrument.
    """
    class Meta:
        name = "virtual-instrument"

    virtual_library: Optional[str] = field(
        default=None,
        metadata={
            "name": "virtual-library",
            "type": "Element",
        }
    )
    virtual_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "virtual-name",
            "type": "Element",
        }
    )


@dataclass
class Wait:
    """The wait type specifies a point where the accompaniment should wait for
    a performer event before continuing.

    This typically happens at the start of new sections or after a held
    note or indeterminate music. These waiting points cannot always be
    inferred reliably from the contents of the displayed score. The
    optional player and time-only attributes restrict the type to apply
    to a single player or set of times through a repeated section,
    respectively.
    """
    class Meta:
        name = "wait"

    player: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )


class WedgeType(Enum):
    """The wedge type is crescendo for the start of a wedge that is closed at
    the left side, diminuendo for the start of a wedge that is closed on the
    right side, and stop for the end of a wedge.

    The continue type is used for formatting wedges over a system break,
    or for other situations where a single wedge is divided into
    multiple segments.
    """
    CRESCENDO = "crescendo"
    DIMINUENDO = "diminuendo"
    STOP = "stop"
    CONTINUE = "continue"


class Winged(Enum):
    """The winged attribute indicates whether the repeat has winged extensions
    that appear above and below the barline.

    The straight and curved values represent single wings, while the
    double-straight and double-curved values represent double wings. The
    none value indicates no wings and is the default.
    """
    NONE = "none"
    STRAIGHT = "straight"
    CURVED = "curved"
    DOUBLE_STRAIGHT = "double-straight"
    DOUBLE_CURVED = "double-curved"


class WoodValue(Enum):
    """The wood-value type represents pictograms for wood percussion
    instruments.

    The maraca and maracas values distinguish the one- and two-maraca
    versions of the pictogram.
    """
    BAMBOO_SCRAPER = "bamboo scraper"
    BOARD_CLAPPER = "board clapper"
    CABASA = "cabasa"
    CASTANETS = "castanets"
    CASTANETS_WITH_HANDLE = "castanets with handle"
    CLAVES = "claves"
    FOOTBALL_RATTLE = "football rattle"
    GUIRO = "guiro"
    LOG_DRUM = "log drum"
    MARACA = "maraca"
    MARACAS = "maracas"
    QUIJADA = "quijada"
    RAINSTICK = "rainstick"
    RATCHET = "ratchet"
    RECO_RECO = "reco-reco"
    SANDPAPER_BLOCKS = "sandpaper blocks"
    SLIT_DRUM = "slit drum"
    TEMPLE_BLOCK = "temple block"
    VIBRASLAP = "vibraslap"
    WHIP = "whip"
    WOOD_BLOCK = "wood block"


class YesNo(Enum):
    """The yes-no type is used for boolean-like attributes.

    We cannot use W3C XML Schema booleans due to their restrictions on
    expression of boolean values.
    """
    YES = "yes"
    NO = "no"


@dataclass
class Accidental:
    """The accidental type represents actual notated accidentals.

    Editorial and cautionary indications are indicated by attributes.
    Values for these attributes are "no" if not present. Specific
    graphic display such as parentheses, brackets, and size are
    controlled by the level-display attribute group.
    """
    class Meta:
        name = "accidental"

    value: Optional[AccidentalValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    cautionary: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    editorial: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    bracket: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    size: Optional[SymbolSize] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"(acc|medRenFla|medRenNatura|medRenShar|kievanAccidental)(\c+)",
        }
    )


@dataclass
class AccidentalMark:
    """An accidental-mark can be used as a separate notation or as part of an
    ornament.

    When used in an ornament, position and placement are relative to the
    ornament, not relative to the note.
    """
    class Meta:
        name = "accidental-mark"

    value: Optional[AccidentalValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    bracket: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    size: Optional[SymbolSize] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"(acc|medRenFla|medRenNatura|medRenShar|kievanAccidental)(\c+)",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class AccidentalText:
    """
    The accidental-text type represents an element with an accidental value and
    text-formatting attributes.
    """
    class Meta:
        name = "accidental-text"

    value: Optional[AccidentalValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    line_height: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "line-height",
            "type": "Attribute",
        }
    )
    lang: Optional[Union[str, LangValue]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    space: Optional[SpaceValue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"(acc|medRenFla|medRenNatura|medRenShar|kievanAccidental)(\c+)",
        }
    )


@dataclass
class Accord:
    """The accord type represents the tuning of a single string in the
    scordatura element.

    It uses the same group of elements as the staff-tuning element.
    Strings are numbered from high to low.

    :ivar tuning_step: The tuning-step element is represented like the
        step element, with a different name to reflect its different
        function in string tuning.
    :ivar tuning_alter: The tuning-alter element is represented like the
        alter element, with a different name to reflect its different
        function in string tuning.
    :ivar tuning_octave: The tuning-octave element is represented like
        the octave element, with a different name to reflect its
        different function in string tuning.
    :ivar string:
    """
    class Meta:
        name = "accord"

    tuning_step: Optional[Step] = field(
        default=None,
        metadata={
            "name": "tuning-step",
            "type": "Element",
            "required": True,
        }
    )
    tuning_alter: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "tuning-alter",
            "type": "Element",
        }
    )
    tuning_octave: Optional[int] = field(
        default=None,
        metadata={
            "name": "tuning-octave",
            "type": "Element",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )
    string: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class AccordionRegistration:
    """The accordion-registration type is used for accordion registration
    symbols.

    These are circular symbols divided horizontally into high, middle,
    and low sections that correspond to 4', 8', and 16' pipes. Each
    accordion-high, accordion-middle, and accordion-low element
    represents the presence of one or more dots in the registration
    diagram. An accordion-registration element needs to have at least
    one of the child elements present.

    :ivar accordion_high: The accordion-high element indicates the
        presence of a dot in the high (4') section of the registration
        symbol. This element is omitted if no dot is present.
    :ivar accordion_middle: The accordion-middle element indicates the
        presence of 1 to 3 dots in the middle (8') section of the
        registration symbol. This element is omitted if no dots are
        present.
    :ivar accordion_low: The accordion-low element indicates the
        presence of a dot in the low (16') section of the registration
        symbol. This element is omitted if no dot is present.
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar id:
    """
    class Meta:
        name = "accordion-registration"

    accordion_high: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "accordion-high",
            "type": "Element",
        }
    )
    accordion_middle: Optional[int] = field(
        default=None,
        metadata={
            "name": "accordion-middle",
            "type": "Element",
            "min_inclusive": 1,
            "max_inclusive": 3,
        }
    )
    accordion_low: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "accordion-low",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Arpeggiate:
    """The arpeggiate type indicates that this note is part of an arpeggiated
    chord.

    The number attribute can be used to distinguish between two
    simultaneous chords arpeggiated separately (different numbers) or
    together (same number). The direction attribute is used if there is
    an arrow on the arpeggio sign. By default, arpeggios go from the
    lowest to highest note.  The length of the sign can be determined
    from the position attributes for the arpeggiate elements used with
    the top and bottom notes of the arpeggiated chord. If the unbroken
    attribute is set to yes, it indicates that the arpeggio continues
    onto another staff within the part. This serves as a hint to
    applications and is not required for cross-staff arpeggios.
    """
    class Meta:
        name = "arpeggiate"

    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    direction: Optional[UpDown] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    unbroken: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Arrow:
    """The arrow element represents an arrow used for a musical technical
    indication.

    It can represent both Unicode and SMuFL arrows. The presence of an
    arrowhead element indicates that only the arrowhead is displayed,
    not the arrow stem. The smufl attribute distinguishes different
    SMuFL glyphs that have an arrow appearance such as arrowBlackUp,
    guitarStrumUp, or handbellsSwingUp. The specified glyph should match
    the descriptive representation.
    """
    class Meta:
        name = "arrow"

    arrow_direction: Optional[ArrowDirection] = field(
        default=None,
        metadata={
            "name": "arrow-direction",
            "type": "Element",
        }
    )
    arrow_style: Optional[ArrowStyle] = field(
        default=None,
        metadata={
            "name": "arrow-style",
            "type": "Element",
        }
    )
    arrowhead: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    circular_arrow: Optional[CircularArrow] = field(
        default=None,
        metadata={
            "name": "circular-arrow",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Assess:
    """By default, an assessment application should assess all notes without a
    cue child element, and not assess any note with a cue child element.

    The assess type allows this default assessment to be overridden for
    individual notes. The optional player and time-only attributes
    restrict the type to apply to a single player or set of times
    through a repeated section, respectively. If missing, the type
    applies to all players or all times through the repeated section,
    respectively. The player attribute references the id attribute of a
    player element defined within the matching score-part.
    """
    class Meta:
        name = "assess"

    type: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    player: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )


@dataclass
class BarStyleColor:
    """
    The bar-style-color type contains barline style and color information.
    """
    class Meta:
        name = "bar-style-color"

    value: Optional[BarStyle] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Barre:
    """The barre element indicates placing a finger over multiple strings on a
    single fret.

    The type is "start" for the lowest pitched string (e.g., the string
    with the highest MusicXML number) and is "stop" for the highest
    pitched string.
    """
    class Meta:
        name = "barre"

    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class BassStep:
    """The bass-step type represents the pitch step of the bass of the current
    chord within the harmony element.

    The text attribute indicates how the bass should appear in a score
    if not using the element contents.
    """
    class Meta:
        name = "bass-step"

    value: Optional[Step] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Beam:
    """Beam values include begin, continue, end, forward hook, and backward
    hook.

    Up to eight concurrent beams are available to cover up to 1024th
    notes. Each beam in a note is represented with a separate beam
    element, starting with the eighth note beam using a number attribute
    of 1. Note that the beam number does not distinguish sets of beams
    that overlap, as it does for slur and other elements. Beaming groups
    are distinguished by being in different voices and/or the presence
    or absence of grace and cue elements. Beams that have a begin value
    can also have a fan attribute to indicate accelerandos and
    ritardandos using fanned beams. The fan attribute may also be used
    with a continue value if the fanning direction changes on that note.
    The value is "none" if not specified. The repeater attribute has
    been deprecated in MusicXML 3.0. Formerly used for tremolos, it
    needs to be specified with a "yes" value for each beam using it.
    """
    class Meta:
        name = "beam"

    value: Optional[BeamValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 8,
        }
    )
    repeater: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    fan: Optional[Fan] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class BeatRepeat:
    """The beat-repeat type is used to indicate that a single beat (but
    possibly many notes) is repeated.

    The slashes attribute specifies the number of slashes to use in the
    symbol. The use-dots attribute indicates whether or not to use dots
    as well (for instance, with mixed rhythm patterns). The value for
    slashes is 1 and the value for use-dots is no if not specified. The
    stop type indicates the first beat where the repeats are no longer
    displayed. Both the start and stop of the beat being repeated should
    be specified unless the repeats are displayed through the end of the
    part. The beat-repeat element specifies a notation style for
    repetitions. The actual music being repeated needs to be repeated
    within the MusicXML file. This element specifies the notation that
    indicates the repeat.

    :ivar slash_type: The slash-type element indicates the graphical
        note type to use for the display of repetition marks.
    :ivar slash_dot: The slash-dot element is used to specify any
        augmentation dots in the note type used to display repetition
        marks.
    :ivar except_voice: The except-voice element is used to specify a
        combination of slash notation and regular notation. Any note
        elements that are in voices specified by the except-voice
        elements are displayed in normal notation, in addition to the
        slash notation that is always displayed.
    :ivar type:
    :ivar slashes:
    :ivar use_dots:
    """
    class Meta:
        name = "beat-repeat"

    slash_type: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "name": "slash-type",
            "type": "Element",
        }
    )
    slash_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "slash-dot",
            "type": "Element",
        }
    )
    except_voice: List[str] = field(
        default_factory=list,
        metadata={
            "name": "except-voice",
            "type": "Element",
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    slashes: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    use_dots: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "use-dots",
            "type": "Attribute",
        }
    )


@dataclass
class BeatUnitTied:
    """The beat-unit-tied type indicates a beat-unit within a metronome mark
    that is tied to the preceding beat-unit.

    This allows two or more tied notes to be associated with a per-
    minute value in a metronome mark, whereas the metronome-tied element
    is restricted to metric relationship marks.

    :ivar beat_unit: The beat-unit element indicates the graphical note
        type to use in a metronome mark.
    :ivar beat_unit_dot: The beat-unit-dot element is used to specify
        any augmentation dots for a metronome mark note.
    """
    class Meta:
        name = "beat-unit-tied"

    beat_unit: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "name": "beat-unit",
            "type": "Element",
            "required": True,
        }
    )
    beat_unit_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "beat-unit-dot",
            "type": "Element",
        }
    )


@dataclass
class Beater:
    """
    The beater type represents pictograms for beaters, mallets, and sticks that
    do not have different materials represented in the pictogram.
    """
    class Meta:
        name = "beater"

    value: Optional[BeaterValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    tip: Optional[TipDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Bracket:
    """Brackets are combined with words in a variety of modern directions.

    The line-end attribute specifies if there is a jog up or down (or
    both), an arrow, or nothing at the start or end of the bracket. If
    the line-end is up or down, the length of the jog can be specified
    using the end-length attribute. The line-type is solid if not
    specified.
    """
    class Meta:
        name = "bracket"

    type: Optional[StartStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line_end: Optional[LineEnd] = field(
        default=None,
        metadata={
            "name": "line-end",
            "type": "Attribute",
            "required": True,
        }
    )
    end_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "end-length",
            "type": "Attribute",
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class BreathMark:
    """
    The breath-mark element indicates a place to take a breath.
    """
    class Meta:
        name = "breath-mark"

    value: Optional[BreathMarkValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Caesura:
    """The caesura element indicates a slight pause.

    It is notated using a "railroad tracks" symbol or other variations
    specified in the element content.
    """
    class Meta:
        name = "caesura"

    value: Optional[CaesuraValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Cancel:
    """A cancel element indicates that the old key signature should be
    cancelled before the new one appears.

    This will always happen when changing to C major or A minor and need
    not be specified then. The cancel value matches the fifths value of
    the cancelled key signature (e.g., a cancel of -2 will provide an
    explicit cancellation for changing from B flat major to F major).
    The optional location attribute indicates where the cancellation
    appears relative to the new key signature.
    """
    class Meta:
        name = "cancel"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    location: Optional[CancelLocation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Clef:
    """Clefs are represented by a combination of sign, line, and clef-octave-
    change elements.

    The optional number attribute refers to staff numbers within the
    part. A value of 1 is assumed if not present. Sometimes clefs are
    added to the staff in non-standard line positions, either to
    indicate cue passages, or when there are multiple clefs present
    simultaneously on one staff. In this situation, the additional
    attribute is set to "yes" and the line value is ignored. The size
    attribute is used for clefs where the additional attribute is "yes".
    It is typically used to indicate cue clefs. Sometimes clefs at the
    start of a measure need to appear after the barline rather than
    before, as for cues or for use after a repeated section. The after-
    barline attribute is set to "yes" in this situation. The attribute
    is ignored for mid-measure clefs. Clefs appear at the start of each
    system unless the print-object attribute has been set to "no" or the
    additional attribute has been set to "yes".

    :ivar sign: The sign element represents the clef symbol.
    :ivar line: Line numbers are counted from the bottom of the staff.
        They are only needed with the G, F, and C signs in order to
        position a pitch correctly on the staff. Standard values are 2
        for the G sign (treble clef), 4 for the F sign (bass clef), and
        3 for the C sign (alto clef). Line values can be used to specify
        positions outside the staff, such as a C clef positioned in the
        middle of a grand staff.
    :ivar clef_octave_change: The clef-octave-change element is used for
        transposing clefs. A treble clef for tenors would have a value
        of -1.
    :ivar number:
    :ivar additional:
    :ivar size:
    :ivar after_barline:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar print_object:
    :ivar id:
    """
    class Meta:
        name = "clef"

    sign: Optional[ClefSign] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    line: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    clef_octave_change: Optional[int] = field(
        default=None,
        metadata={
            "name": "clef-octave-change",
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    additional: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    size: Optional[SymbolSize] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    after_barline: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "after-barline",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Coda:
    """The coda type is the visual indicator of a coda sign.

    The exact glyph can be specified with the smufl attribute. A sound
    element is also needed to guide playback applications reliably.
    """
    class Meta:
        name = "coda"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"coda\c*",
        }
    )


@dataclass
class Dashes:
    """The dashes type represents dashes, used for instance with cresc.

    and dim. marks.
    """
    class Meta:
        name = "dashes"

    type: Optional[StartStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class DegreeAlter:
    """The degree-alter type represents the chromatic alteration for the
    current degree.

    If the degree-type value is alter or subtract, the degree-alter
    value is relative to the degree already in the chord based on its
    kind element. If the degree-type value is add, the degree-alter is
    relative to a dominant chord (major and perfect intervals except for
    a minor seventh). The plus-minus attribute is used to indicate if
    plus and minus symbols should be used instead of sharp and flat
    symbols to display the degree alteration. It is no if not specified.
    """
    class Meta:
        name = "degree-alter"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    plus_minus: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "plus-minus",
            "type": "Attribute",
        }
    )


@dataclass
class DegreeType:
    """The degree-type type indicates if this degree is an addition,
    alteration, or subtraction relative to the kind of the current chord.

    The value of the degree-type element affects the interpretation of
    the value of the degree-alter element. The text attribute specifies
    how the type of the degree should be displayed.
    """
    class Meta:
        name = "degree-type"

    value: Optional[DegreeTypeValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class DegreeValue:
    """The content of the degree-value type is a number indicating the degree
    of the chord (1 for the root, 3 for third, etc).

    The text attribute specifies how the value of the degree should be
    displayed. The symbol attribute indicates that a symbol should be
    used in specifying the degree. If the symbol attribute is present,
    the value of the text attribute follows the symbol.
    """
    class Meta:
        name = "degree-value"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    symbol: Optional[DegreeSymbolValue] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Double:
    """The double type indicates that the music is doubled one octave from what
    is currently written.

    If the above attribute is set to yes, the doubling is one octave
    above what is written, as for mixed flute / piccolo parts in band
    literature. Otherwise the doubling is one octave below what is
    written, as for mixed cello / bass parts in orchestral literature.
    """
    class Meta:
        name = "double"

    above: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Dynamics:
    """Dynamics can be associated either with a note or a general musical
    direction.

    To avoid inconsistencies between and amongst the letter
    abbreviations for dynamics (what is sf vs. sfz, standing alone or
    with a trailing dynamic that is not always piano), we use the actual
    letters as the names of these dynamic elements. The other-dynamics
    element allows other dynamic marks that are not covered here.
    Dynamics elements may also be combined to create marks not covered
    by a single element, such as sfmp. These letter dynamic symbols are
    separated from crescendo, decrescendo, and wedge indications.
    Dynamic representation is inconsistent in scores. Many things are
    assumed by the composer and left out, such as returns to original
    dynamics. The MusicXML format captures what is in the score, but
    does not try to be optimal for analysis or synthesis of dynamics.
    The placement attribute is used when the dynamics are associated
    with a note. It is ignored when the dynamics are associated with a
    direction. In that case the direction element's placement attribute
    is used instead.
    """
    class Meta:
        name = "dynamics"

    p: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    pp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    ppp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    pppp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    ppppp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    pppppp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    f: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    ff: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    fff: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    ffff: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    fffff: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    ffffff: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    mp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    mf: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sf: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sfp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sfpp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    fp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    rf: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    rfz: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sfz: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sffz: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    fz: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    n: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    pf: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    sfzp: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    other_dynamics: List[OtherText] = field(
        default_factory=list,
        metadata={
            "name": "other-dynamics",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Effect:
    """The effect type represents pictograms for sound effect percussion
    instruments.

    The smufl attribute is used to distinguish different SMuFL stylistic
    alternates.
    """
    class Meta:
        name = "effect"

    value: Optional[EffectValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class Elision:
    """The elision type represents an elision between lyric syllables.

    The text content specifies the symbol used to display the elision.
    Common values are a no-break space (Unicode 00A0), an underscore
    (Unicode 005F), or an undertie (Unicode 203F). If the text content
    is empty, the smufl attribute is used to specify the symbol to use.
    Its value is a SMuFL canonical glyph name that starts with lyrics.
    The SMuFL attribute is ignored if the elision glyph is already
    specified by the text content. If neither text content nor a smufl
    attribute are present, the elision glyph is application-specific.
    """
    class Meta:
        name = "elision"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"lyrics\c+",
        }
    )


@dataclass
class EmptyFont:
    """
    The empty-font type represents an empty element with font attributes.
    """
    class Meta:
        name = "empty-font"

    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )


@dataclass
class EmptyLine:
    """
    The empty-line type represents an empty element with line-shape, line-type,
    line-length, dashed-formatting, print-style and placement attributes.
    """
    class Meta:
        name = "empty-line"

    line_shape: Optional[LineShape] = field(
        default=None,
        metadata={
            "name": "line-shape",
            "type": "Attribute",
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    line_length: Optional[LineLength] = field(
        default=None,
        metadata={
            "name": "line-length",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyPlacement:
    """
    The empty-placement type represents an empty element with print-style and
    placement attributes.
    """
    class Meta:
        name = "empty-placement"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyPlacementSmufl:
    """
    The empty-placement-smufl type represents an empty element with print-
    style, placement, and smufl attributes.
    """
    class Meta:
        name = "empty-placement-smufl"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyPrintObjectStyleAlign:
    """
    The empty-print-style-align-object type represents an empty element with
    print-object and print-style-align attribute groups.
    """
    class Meta:
        name = "empty-print-object-style-align"

    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyPrintStyle:
    """
    The empty-print-style type represents an empty element with print-style
    attribute group.
    """
    class Meta:
        name = "empty-print-style"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class EmptyPrintStyleAlign:
    """
    The empty-print-style-align type represents an empty element with print-
    style-align attribute group.
    """
    class Meta:
        name = "empty-print-style-align"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyPrintStyleAlignId:
    """
    The empty-print-style-align-id type represents an empty element with print-
    style-align and optional-unique-id attribute groups.
    """
    class Meta:
        name = "empty-print-style-align-id"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class EmptyTrillSound:
    """
    The empty-trill-sound type represents an empty element with print-style,
    placement, and trill-sound attributes.
    """
    class Meta:
        name = "empty-trill-sound"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    start_note: Optional[StartNote] = field(
        default=None,
        metadata={
            "name": "start-note",
            "type": "Attribute",
        }
    )
    trill_step: Optional[TrillStep] = field(
        default=None,
        metadata={
            "name": "trill-step",
            "type": "Attribute",
        }
    )
    two_note_turn: Optional[TwoNoteTurn] = field(
        default=None,
        metadata={
            "name": "two-note-turn",
            "type": "Attribute",
        }
    )
    accelerate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    beats: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("2"),
        }
    )
    second_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "second-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    last_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "last-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )


@dataclass
class Ending:
    """The ending type represents multiple (e.g. first and second) endings.

    Typically, the start type is associated with the left barline of the
    first measure in an ending. The stop and discontinue types are
    associated with the right barline of the last measure in an ending.
    Stop is used when the ending mark concludes with a downward jog, as
    is typical for first endings. Discontinue is used when there is no
    downward jog, as is typical for second endings that do not conclude
    a piece. The length of the jog can be specified using the end-length
    attribute. The text-x and text-y attributes are offsets that specify
    where the baseline of the start of the ending text appears, relative
    to the start of the ending line. The number attribute indicates
    which times the ending is played, similar to the time-only attribute
    used by other elements. While this often represents the numeric
    values for what is under the ending line, it can also indicate
    whether an ending is played during a larger dal segno or da capo
    repeat. Single endings such as "1" or comma-separated multiple
    endings such as "1,2" may be used. The ending element text is used
    when the text displayed in the ending is different than what appears
    in the number attribute. The print-object attribute is used to
    indicate when an ending is present but not printed, as is often the
    case for many parts in a full score.
    """
    class Meta:
        name = "ending"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    number: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
            "pattern": r"([ ]*)|([1-9][0-9]*(, ?[1-9][0-9]*)*)",
        }
    )
    type: Optional[StartStopDiscontinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    system: Optional[SystemRelation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    end_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "end-length",
            "type": "Attribute",
        }
    )
    text_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "text-x",
            "type": "Attribute",
        }
    )
    text_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "text-y",
            "type": "Attribute",
        }
    )


@dataclass
class Extend:
    """The extend type represents lyric word extension / melisma lines as well
    as figured bass extensions.

    The optional type and position attributes are added in Version 3.0
    to provide better formatting control.
    """
    class Meta:
        name = "extend"

    type: Optional[StartStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Fermata:
    """The fermata text content represents the shape of the fermata sign.

    An empty fermata element represents a normal fermata. The fermata
    type is upright if not specified.
    """
    class Meta:
        name = "fermata"

    value: Optional[FermataShape] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    type: Optional[UprightInverted] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Fingering:
    """Fingering is typically indicated 1,2,3,4,5.

    Multiple fingerings may be given, typically to substitute fingerings
    in the middle of a note. The substitution and alternate values are
    "no" if the attribute is not present. For guitar and other fretted
    instruments, the fingering element represents the fretting finger;
    the pluck element represents the plucking finger.
    """
    class Meta:
        name = "fingering"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    substitution: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    alternate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FirstFret:
    """The first-fret type indicates which fret is shown in the top space of
    the frame; it is fret 1 if the element is not present.

    The optional text attribute indicates how this is represented in the
    fret diagram, while the location attribute indicates whether the
    text appears to the left or right of the frame.
    """
    class Meta:
        name = "first-fret"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    location: Optional[LeftRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FormattedSymbol:
    """
    The formatted-symbol type represents a SMuFL musical symbol element with
    formatting attributes.
    """
    class Meta:
        name = "formatted-symbol"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    line_height: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "line-height",
            "type": "Attribute",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FormattedSymbolId:
    """
    The formatted-symbol-id type represents a SMuFL musical symbol element with
    formatting and id attributes.
    """
    class Meta:
        name = "formatted-symbol-id"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    line_height: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "line-height",
            "type": "Attribute",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FormattedText:
    """
    The formatted-text type represents a text element with text-formatting
    attributes.
    """
    class Meta:
        name = "formatted-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    line_height: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "line-height",
            "type": "Attribute",
        }
    )
    lang: Optional[Union[str, LangValue]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    space: Optional[SpaceValue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FormattedTextId:
    """
    The formatted-text-id type represents a text element with text-formatting
    and id attributes.
    """
    class Meta:
        name = "formatted-text-id"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    line_height: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "line-height",
            "type": "Attribute",
        }
    )
    lang: Optional[Union[str, LangValue]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    space: Optional[SpaceValue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Fret:
    """The fret element is used with tablature notation and chord diagrams.

    Fret numbers start with 0 for an open string and 1 for the first
    fret.
    """
    class Meta:
        name = "fret"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Glass:
    """The glass type represents pictograms for glass percussion instruments.

    The smufl attribute is used to distinguish different SMuFL glyphs
    for wind chimes in the Chimes pictograms range, including those made
    of materials other than glass.
    """
    class Meta:
        name = "glass"

    value: Optional[GlassValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class Glissando:
    """Glissando and slide types both indicate rapidly moving from one pitch to
    the other so that individual notes are not discerned.

    A glissando sounds the distinct notes in between the two pitches and
    defaults to a wavy line. The optional text is printed alongside the
    line.
    """
    class Meta:
        name = "glissando"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Grace:
    """The grace type indicates the presence of a grace note.

    The slash attribute for a grace note is yes for slashed grace notes.
    The steal-time-previous attribute indicates the percentage of time
    to steal from the previous note for the grace note. The steal-time-
    following attribute indicates the percentage of time to steal from
    the following note for the grace note, as for appoggiaturas. The
    make-time attribute indicates to make time, not steal time; the
    units are in real-time divisions for the grace note.
    """
    class Meta:
        name = "grace"

    steal_time_previous: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "steal-time-previous",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    steal_time_following: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "steal-time-following",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    make_time: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "make-time",
            "type": "Attribute",
        }
    )
    slash: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class GroupBarline:
    """
    The group-barline type indicates if the group should have common barlines.
    """
    class Meta:
        name = "group-barline"

    value: Optional[GroupBarlineValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class GroupName:
    """The group-name type describes the name or abbreviation of a part-group
    element.

    Formatting attributes in the group-name type are deprecated in
    Version 2.0 in favor of the new group-name-display and group-
    abbreviation-display elements.
    """
    class Meta:
        name = "group-name"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class GroupSymbol:
    """The group-symbol type indicates how the symbol for a group is indicated
    in the score.

    It is none if not specified.
    """
    class Meta:
        name = "group-symbol"

    value: Optional[GroupSymbolValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Grouping:
    """The grouping type is used for musical analysis.

    When the type attribute is "start" or "single", it usually contains
    one or more feature elements. The number attribute is used for
    distinguishing between overlapping and hierarchical groupings. The
    member-of attribute allows for easy distinguishing of what grouping
    elements are in what hierarchy. Feature elements contained within a
    "stop" type of grouping may be ignored. This element is flexible to
    allow for different types of analyses. Future versions of the
    MusicXML format may add elements that can represent more
    standardized categories of analysis data, allowing for easier data
    sharing.
    """
    class Meta:
        name = "grouping"

    feature: List[Feature] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    type: Optional[StartStopSingle] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: str = field(
        default="1",
        metadata={
            "type": "Attribute",
        }
    )
    member_of: Optional[str] = field(
        default=None,
        metadata={
            "name": "member-of",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HammerOnPullOff:
    """The hammer-on and pull-off elements are used in guitar and fretted
    instrument notation.

    Since a single slur can be marked over many notes, the hammer-on and
    pull-off elements are separate so the individual pair of notes can
    be specified. The element content can be used to specify how the
    hammer-on or pull-off should be notated. An empty element leaves
    this choice up to the application.
    """
    class Meta:
        name = "hammer-on-pull-off"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Handbell:
    """
    The handbell element represents notation for various techniques used in
    handbell and handchime music.
    """
    class Meta:
        name = "handbell"

    value: Optional[HandbellValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HarmonClosed:
    """The harmon-closed type represents whether the harmon mute is closed,
    open, or half-open.

    The optional location attribute indicates which portion of the
    symbol is filled in when the element value is half.
    """
    class Meta:
        name = "harmon-closed"

    value: Optional[HarmonClosedValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    location: Optional[HarmonClosedLocation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Harmonic:
    """The harmonic type indicates natural and artificial harmonics.

    Allowing the type of pitch to be specified, combined with controls
    for appearance/playback differences, allows both the notation and
    the sound to be represented. Artificial harmonics can add a notated
    touching pitch; artificial pinch harmonics will usually not notate a
    touching pitch. The attributes for the harmonic element refer to the
    use of the circular harmonic symbol, typically but not always used
    with natural harmonics.

    :ivar natural: The natural element indicates that this is a natural
        harmonic. These are usually notated at base pitch rather than
        sounding pitch.
    :ivar artificial: The artificial element indicates that this is an
        artificial harmonic.
    :ivar base_pitch: The base pitch is the pitch at which the string is
        played before touching to create the harmonic.
    :ivar touching_pitch: The touching-pitch is the pitch at which the
        string is touched lightly to produce the harmonic.
    :ivar sounding_pitch: The sounding-pitch is the pitch which is heard
        when playing the harmonic.
    :ivar print_object:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar placement:
    """
    class Meta:
        name = "harmonic"

    natural: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    artificial: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    base_pitch: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "base-pitch",
            "type": "Element",
        }
    )
    touching_pitch: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "touching-pitch",
            "type": "Element",
        }
    )
    sounding_pitch: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "sounding-pitch",
            "type": "Element",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HarmonyAlter:
    """The harmony-alter type represents the chromatic alteration of the root,
    numeral, or bass of the current harmony-chord group within the harmony
    element.

    In some chord styles, the text of the preceding element may include
    alteration information. In that case, the print-object attribute of
    this type can be set to no. The location attribute indicates whether
    the alteration should appear to the left or the right of the
    preceding element. Its default value varies by element.
    """
    class Meta:
        name = "harmony-alter"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    location: Optional[LeftRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HoleClosed:
    """The hole-closed type represents whether the hole is closed, open, or
    half-open.

    The optional location attribute indicates which portion of the hole
    is filled in when the element value is half.
    """
    class Meta:
        name = "hole-closed"

    value: Optional[HoleClosedValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    location: Optional[HoleClosedLocation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HorizontalTurn:
    """The horizontal-turn type represents turn elements that are horizontal
    rather than vertical.

    These are empty elements with print-style, placement, trill-sound,
    and slash attributes. If the slash attribute is yes, then a vertical
    line is used to slash the turn. It is no if not specified.
    """
    class Meta:
        name = "horizontal-turn"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    start_note: Optional[StartNote] = field(
        default=None,
        metadata={
            "name": "start-note",
            "type": "Attribute",
        }
    )
    trill_step: Optional[TrillStep] = field(
        default=None,
        metadata={
            "name": "trill-step",
            "type": "Attribute",
        }
    )
    two_note_turn: Optional[TwoNoteTurn] = field(
        default=None,
        metadata={
            "name": "two-note-turn",
            "type": "Attribute",
        }
    )
    accelerate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    beats: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("2"),
        }
    )
    second_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "second-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    last_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "last-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    slash: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Image:
    """
    The image type is used to include graphical images in a score.
    """
    class Meta:
        name = "image"

    source: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    type: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    height: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    width: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[ValignImage] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class InstrumentChange:
    """The instrument-change element type represents a change to the virtual
    instrument sound for a given score-instrument.

    The id attribute refers to the score-instrument affected by the
    change. All instrument-change child elements can also be initially
    specified within the score-instrument element.

    :ivar instrument_sound: The instrument-sound element describes the
        default timbre of the score-instrument. This description is
        independent of a particular virtual or MIDI instrument
        specification and allows playback to be shared more easily
        between applications and libraries.
    :ivar solo: The solo element is present if performance is intended
        by a solo instrument.
    :ivar ensemble: The ensemble element is present if performance is
        intended by an ensemble such as an orchestral section. The text
        of the ensemble element contains the size of the section, or is
        empty if the ensemble size is not specified.
    :ivar virtual_instrument:
    :ivar id:
    """
    class Meta:
        name = "instrument-change"

    instrument_sound: Optional[str] = field(
        default=None,
        metadata={
            "name": "instrument-sound",
            "type": "Element",
        }
    )
    solo: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    ensemble: Optional[Union[int, PositiveIntegerOrEmptyValue]] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    virtual_instrument: Optional[VirtualInstrument] = field(
        default=None,
        metadata={
            "name": "virtual-instrument",
            "type": "Element",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class Interchangeable:
    """The interchangeable type is used to represent the second in a pair of
    interchangeable dual time signatures, such as the 6/8 in 3/4 (6/8).

    A separate symbol attribute value is available compared to the time
    element's symbol attribute, which applies to the first of the dual
    time signatures.

    :ivar time_relation:
    :ivar beats: The beats element indicates the number of beats, as
        found in the numerator of a time signature.
    :ivar beat_type: The beat-type element indicates the beat unit, as
        found in the denominator of a time signature.
    :ivar symbol:
    :ivar separator:
    """
    class Meta:
        name = "interchangeable"

    time_relation: Optional[TimeRelation] = field(
        default=None,
        metadata={
            "name": "time-relation",
            "type": "Element",
        }
    )
    beats: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequential": True,
        }
    )
    beat_type: List[str] = field(
        default_factory=list,
        metadata={
            "name": "beat-type",
            "type": "Element",
            "min_occurs": 1,
            "sequential": True,
        }
    )
    symbol: Optional[TimeSymbol] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    separator: Optional[TimeSeparator] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Inversion:
    """The inversion type represents harmony inversions.

    The value is a number indicating which inversion is used: 0 for root
    position, 1 for first inversion, etc.  The text attribute indicates
    how the inversion should be displayed in a score.
    """
    class Meta:
        name = "inversion"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class KeyAccidental:
    """
    The key-accidental type indicates the accidental to be displayed in a non-
    traditional key signature, represented in the same manner as the accidental
    type without the formatting attributes.
    """
    class Meta:
        name = "key-accidental"

    value: Optional[AccidentalValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"(acc|medRenFla|medRenNatura|medRenShar|kievanAccidental)(\c+)",
        }
    )


@dataclass
class KeyOctave:
    """The key-octave type specifies in which octave an element of a key
    signature appears.

    The content specifies the octave value using the same values as the
    display-octave element. The number attribute is a positive integer
    that refers to the key signature element in left-to-right order. If
    the cancel attribute is set to yes, then this number refers to the
    canceling key signature specified by the cancel element in the
    parent key element. The cancel attribute cannot be set to yes if
    there is no corresponding cancel element within the parent key
    element. It is no by default.
    """
    class Meta:
        name = "key-octave"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    cancel: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Kind:
    """Kind indicates the type of chord.

    Degree elements can then add, subtract, or alter from these starting points
    The attributes are used to indicate the formatting of the symbol. Since the kind element is the constant in all the harmony-chord groups that can make up a polychord, many formatting attributes are here.
    The use-symbols attribute is yes if the kind should be represented when possible with harmony symbols rather than letters and numbers. These symbols include:
    major: a triangle, like Unicode 25B3
    minor: -, like Unicode 002D
    augmented: +, like Unicode 002B
    diminished: °, like Unicode 00B0
    half-diminished: ø, like Unicode 00F8
    For the major-minor kind, only the minor symbol is used when use-symbols is yes. The major symbol is set using the symbol attribute in the degree-value element. The corresponding degree-alter value will usually be 0 in this case.
    The text attribute describes how the kind should be spelled in a score. If use-symbols is yes, the value of the text attribute follows the symbol. The stack-degrees attribute is yes if the degree elements should be stacked above each other. The parentheses-degrees attribute is yes if all the degrees should be in parentheses. The bracket-degrees attribute is yes if all the degrees should be in a bracket. If not specified, these values are implementation-specific. The alignment attributes are for the entire harmony-chord group of which this kind element is a part.
    The text attribute may use strings such as "13sus" that refer to both the kind and one or more degree elements. In this case, the corresponding degree elements should have the print-object attribute set to "no" to keep redundant alterations from being displayed.
    """
    class Meta:
        name = "kind"

    value: Optional[KindValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    use_symbols: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "use-symbols",
            "type": "Attribute",
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    stack_degrees: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "stack-degrees",
            "type": "Attribute",
        }
    )
    parentheses_degrees: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "parentheses-degrees",
            "type": "Attribute",
        }
    )
    bracket_degrees: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "bracket-degrees",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Level:
    """The level type is used to specify editorial information for different
    MusicXML elements.

    The content contains identifying and/or descriptive text about the
    editorial status of the parent element. If the reference attribute
    is yes, this indicates editorial information that is for display
    only and should not affect playback. For instance, a modern edition
    of older music may set reference="yes" on the attributes containing
    the music's original clef, key, and time signature. It is no if not
    specified. The type attribute indicates whether the editorial
    information applies to the start of a series of symbols, the end of
    a series of symbols, or a single symbol. It is single if not
    specified for compatibility with earlier MusicXML versions.
    """
    class Meta:
        name = "level"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    reference: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    type: Optional[StartStopSingle] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    bracket: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    size: Optional[SymbolSize] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class LineDetail:
    """If the staff-lines element is present, the appearance of each line may
    be individually specified with a line-detail type.

    Staff lines are numbered from bottom to top. The print-object
    attribute allows lines to be hidden within a staff. This is used in
    special situations such as a widely-spaced percussion staff where a
    note placed below the higher line is distinct from a note placed
    above the lower line. Hidden staff lines are included when
    specifying clef lines and determining display-step / display-octave
    values, but are not counted as lines for the purposes of the system-
    layout and staff-layout elements.
    """
    class Meta:
        name = "line-detail"

    line: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    width: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )


@dataclass
class Link:
    """The link type serves as an outgoing simple XLink.

    If a relative link is used within a document that is part of a
    compressed MusicXML file, the link is relative to the root folder of
    the zip file.
    """
    class Meta:
        name = "link"

    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
            "required": True,
        }
    )
    type: TypeValue = field(
        init=False,
        default=TypeValue.SIMPLE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    show: ShowValue = field(
        default=ShowValue.REPLACE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    actuate: ActuateValue = field(
        default=ActuateValue.ON_REQUEST,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    element: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    position: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )


@dataclass
class LyricFont:
    """
    The lyric-font type specifies the default font for a particular name and
    number of lyric.
    """
    class Meta:
        name = "lyric-font"

    number: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )


@dataclass
class LyricLanguage:
    """
    The lyric-language type specifies the default language for a particular
    name and number of lyric.
    """
    class Meta:
        name = "lyric-language"

    number: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    lang: Optional[Union[str, LangValue]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
            "required": True,
        }
    )


@dataclass
class MeasureNumbering:
    """The measure-numbering type describes how frequently measure numbers are
    displayed on this part.

    The text attribute from the measure element is used for display, or
    the number attribute if the text attribute is not present. Measures
    with an implicit attribute set to "yes" never display a measure
    number, regardless of the measure-numbering setting. The optional
    staff attribute refers to staff numbers within the part, from top to
    bottom on the system. It indicates which staff is used as the
    reference point for vertical positioning. A value of 1 is assumed if
    not present. The optional multiple-rest-always and multiple-rest-
    range attributes describe how measure numbers are shown on multiple
    rests when the measure-numbering value is not set to none. The
    multiple-rest-always attribute is set to yes when the measure number
    should always be shown, even if the multiple rest starts midway
    through a system when measure numbering is set to system level. The
    multiple-rest-range attribute is set to yes when measure numbers on
    multiple rests display the range of numbers for the first and last
    measure, rather than just the number of the first measure.
    """
    class Meta:
        name = "measure-numbering"

    value: Optional[MeasureNumberingValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    system: Optional[SystemRelationNumber] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    staff: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    multiple_rest_always: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "multiple-rest-always",
            "type": "Attribute",
        }
    )
    multiple_rest_range: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "multiple-rest-range",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class MeasureRepeat:
    """The measure-repeat type is used for both single and multiple measure
    repeats.

    The text of the element indicates the number of measures to be
    repeated in a single pattern. The slashes attribute specifies the
    number of slashes to use in the repeat sign. It is 1 if not
    specified. The text of the element is ignored when the type is stop.
    The stop type indicates the first measure where the repeats are no
    longer displayed. Both the start and the stop of the measure-repeat
    should be specified unless the repeats are displayed through the end
    of the part. The measure-repeat element specifies a notation style
    for repetitions. The actual music being repeated needs to be
    repeated within each measure of the MusicXML file. This element
    specifies the notation that indicates the repeat.
    """
    class Meta:
        name = "measure-repeat"

    value: Optional[Union[int, PositiveIntegerOrEmptyValue]] = field(
        default=None
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    slashes: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Membrane:
    """The membrane type represents pictograms for membrane percussion
    instruments.

    The smufl attribute is used to distinguish different SMuFL stylistic
    alternates.
    """
    class Meta:
        name = "membrane"

    value: Optional[MembraneValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class Metal:
    """The metal type represents pictograms for metal percussion instruments.

    The smufl attribute is used to distinguish different SMuFL stylistic
    alternates.
    """
    class Meta:
        name = "metal"

    value: Optional[MetalValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class MetronomeBeam:
    """
    The metronome-beam type works like the beam type in defining metric
    relationships, but does not include all the attributes available in the
    beam type.
    """
    class Meta:
        name = "metronome-beam"

    value: Optional[BeamValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 8,
        }
    )


@dataclass
class MetronomeTied:
    """The metronome-tied indicates the presence of a tie within a metric
    relationship mark.

    As with the tied element, both the start and stop of the tie should
    be specified, in this case within separate metronome-note elements.
    """
    class Meta:
        name = "metronome-tied"

    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class Miscellaneous:
    """If a program has other metadata not yet supported in the MusicXML
    format, it can go in the miscellaneous element.

    The miscellaneous type puts each separate part of metadata into its
    own miscellaneous-field type.
    """
    class Meta:
        name = "miscellaneous"

    miscellaneous_field: List[MiscellaneousField] = field(
        default_factory=list,
        metadata={
            "name": "miscellaneous-field",
            "type": "Element",
        }
    )


@dataclass
class MultipleRest:
    """The text of the multiple-rest type indicates the number of measures in
    the multiple rest.

    Multiple rests may use the 1-bar / 2-bar / 4-bar rest symbols, or a
    single shape. The use-symbols attribute indicates which to use; it
    is no if not specified.
    """
    class Meta:
        name = "multiple-rest"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    use_symbols: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "use-symbols",
            "type": "Attribute",
        }
    )


@dataclass
class NonArpeggiate:
    """The non-arpeggiate type indicates that this note is at the top or bottom
    of a bracket indicating to not arpeggiate these notes.

    Since this does not involve playback, it is only used on the top or
    bottom notes, not on each note as for the arpeggiate type.
    """
    class Meta:
        name = "non-arpeggiate"

    type: Optional[TopBottom] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class NoteSize:
    """The note-size type indicates the percentage of the regular note size to
    use for notes with a cue and large size as defined in the type element.

    The grace type is used for notes of cue size that that include a
    grace element. The cue type is used for all other notes with cue
    size, whether defined explicitly or implicitly via a cue element.
    The large type is used for notes of large size. The text content
    represent the numeric percentage. A value of 100 would be identical
    to the size of a regular note as defined by the music font.
    """
    class Meta:
        name = "note-size"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
            "min_inclusive": Decimal("0"),
        }
    )
    type: Optional[NoteSizeType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class NoteType:
    """The note-type type indicates the graphic note type.

    Values range from 1024th to maxima. The size attribute indicates
    full, cue, grace-cue, or large size. The default is full for regular
    notes, grace-cue for notes that contain both grace and cue elements,
    and cue for notes that contain either a cue or a grace element, but
    not both.
    """
    class Meta:
        name = "note-type"

    value: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    size: Optional[SymbolSize] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Notehead:
    """The notehead type indicates shapes other than the open and closed ovals
    associated with note durations.

    The smufl attribute can be used to specify a particular notehead,
    allowing application interoperability without requiring every SMuFL
    glyph to have a MusicXML element equivalent. This attribute can be
    used either with the "other" value, or to refine a specific notehead
    value such as "cluster". Noteheads in the SMuFL Note name noteheads
    and Note name noteheads supplement ranges (U+E150–U+E1AF and
    U+EEE0–U+EEFF) should not use the smufl attribute or the "other"
    value, but instead use the notehead-text element. For the enclosed
    shapes, the default is to be hollow for half notes and longer, and
    filled otherwise. The filled attribute can be set to change this if
    needed. If the parentheses attribute is set to yes, the notehead is
    parenthesized. It is no by default.
    """
    class Meta:
        name = "notehead"

    value: Optional[NoteheadValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    filled: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class NumeralKey:
    """The numeral-key type is used when the key for the numeral is different
    than the key specified by the key signature.

    The numeral-fifths element specifies the key in the same way as the
    fifths element. The numeral-mode element specifies the mode similar
    to the mode element, but with a restricted set of values
    """
    class Meta:
        name = "numeral-key"

    numeral_fifths: Optional[int] = field(
        default=None,
        metadata={
            "name": "numeral-fifths",
            "type": "Element",
            "required": True,
        }
    )
    numeral_mode: Optional[NumeralMode] = field(
        default=None,
        metadata={
            "name": "numeral-mode",
            "type": "Element",
            "required": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )


@dataclass
class NumeralRoot:
    """The numeral-root type represents the Roman numeral or Nashville number
    as a positive integer from 1 to 7.

    The text attribute indicates how the numeral should appear in the
    score. A numeral-root value of 5 with a kind of major would have a
    text attribute of "V" if displayed as a Roman numeral, and "5" if
    displayed as a Nashville number. If the text attribute is not
    specified, the display is application-dependent.
    """
    class Meta:
        name = "numeral-root"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
            "min_inclusive": 1,
            "max_inclusive": 7,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class OctaveShift:
    """The octave shift type indicates where notes are shifted up or down from
    their true pitched values because of printing difficulty.

    Thus a treble clef line noted with 8va will be indicated with an
    octave-shift down from the pitch data indicated in the notes. A size
    of 8 indicates one octave; a size of 15 indicates two octaves.
    """
    class Meta:
        name = "octave-shift"

    type: Optional[UpDownStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    size: int = field(
        default=8,
        metadata={
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Offset:
    """An offset is represented in terms of divisions, and indicates where the
    direction will appear relative to the current musical location.

    The current musical location is always within the current measure,
    even at the end of a measure. The offset affects the visual
    appearance of the direction. If the sound attribute is "yes", then
    the offset affects playback and listening too. If the sound
    attribute is "no", then any sound or listening associated with the
    direction takes effect at the current location. The sound attribute
    is "no" by default for compatibility with earlier versions of the
    MusicXML format. If an element within a direction includes a
    default-x attribute, the offset value will be ignored when
    determining the appearance of that element.
    """
    class Meta:
        name = "offset"

    value: Optional[Decimal] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    sound: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Opus:
    """
    The opus type represents a link to a MusicXML opus document that composes
    multiple MusicXML scores into a collection.
    """
    class Meta:
        name = "opus"

    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
            "required": True,
        }
    )
    type: TypeValue = field(
        init=False,
        default=TypeValue.SIMPLE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    show: ShowValue = field(
        default=ShowValue.REPLACE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    actuate: ActuateValue = field(
        default=ActuateValue.ON_REQUEST,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )


@dataclass
class OtherDirection:
    """The other-direction type is used to define any direction symbols not yet
    in the MusicXML format.

    The smufl attribute can be used to specify a particular direction
    symbol, allowing application interoperability without requiring
    every SMuFL glyph to have a MusicXML element equivalent. Using the
    other-direction type without the smufl attribute allows for extended
    representation, though without application interoperability.
    """
    class Meta:
        name = "other-direction"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class OtherNotation:
    """The other-notation type is used to define any notations not yet in the
    MusicXML format.

    It handles notations where more specific extension elements such as
    other-dynamics and other-technical are not appropriate. The smufl
    attribute can be used to specify a particular notation, allowing
    application interoperability without requiring every SMuFL glyph to
    have a MusicXML element equivalent. Using the other-notation type
    without the smufl attribute allows for extended representation,
    though without application interoperability.
    """
    class Meta:
        name = "other-notation"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[StartStopSingle] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class OtherPlacementText:
    """The other-placement-text type represents a text element with print-
    style, placement, and smufl attribute groups.

    This type is used by MusicXML notation extension elements to allow
    specification of specific SMuFL glyphs without needed to add every
    glyph as a MusicXML element.
    """
    class Meta:
        name = "other-placement-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PageMargins:
    """Page margins are specified either for both even and odd pages, or via
    separate odd and even page number values.

    The type attribute is not needed when used as part of a print
    element. If omitted when the page-margins type is used in the
    defaults element, "both" is the default value.
    """
    class Meta:
        name = "page-margins"

    left_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "left-margin",
            "type": "Element",
            "required": True,
        }
    )
    right_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "right-margin",
            "type": "Element",
            "required": True,
        }
    )
    top_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "top-margin",
            "type": "Element",
            "required": True,
        }
    )
    bottom_margin: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bottom-margin",
            "type": "Element",
            "required": True,
        }
    )
    type: Optional[MarginType] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PartClef:
    """The child elements of the part-clef type have the same meaning as for
    the clef type.

    However that meaning applies to a transposed part created from the
    existing score file.

    :ivar sign: The sign element represents the clef symbol.
    :ivar line: Line numbers are counted from the bottom of the staff.
        They are only needed with the G, F, and C signs in order to
        position a pitch correctly on the staff. Standard values are 2
        for the G sign (treble clef), 4 for the F sign (bass clef), and
        3 for the C sign (alto clef). Line values can be used to specify
        positions outside the staff, such as a C clef positioned in the
        middle of a grand staff.
    :ivar clef_octave_change: The clef-octave-change element is used for
        transposing clefs. A treble clef for tenors would have a value
        of -1.
    """
    class Meta:
        name = "part-clef"

    sign: Optional[ClefSign] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    line: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    clef_octave_change: Optional[int] = field(
        default=None,
        metadata={
            "name": "clef-octave-change",
            "type": "Element",
        }
    )


@dataclass
class PartLink:
    """The part-link type allows MusicXML data for both score and parts to be
    contained within a single compressed MusicXML file.

    It links a score-part from a score document to MusicXML documents
    that contain parts data. In the case of a single compressed MusicXML
    file, the link href values are paths that are relative to the root
    folder of the zip file.

    :ivar instrument_link:
    :ivar group_link: Multiple part-link elements can reference
        different types of linked documents, such as parts and condensed
        score. The optional group-link elements identify the groups used
        in the linked document. The content of a group-link element
        should match the content of a group element in the linked
        document.
    :ivar href:
    :ivar type:
    :ivar role:
    :ivar title:
    :ivar show:
    :ivar actuate:
    """
    class Meta:
        name = "part-link"

    instrument_link: List[InstrumentLink] = field(
        default_factory=list,
        metadata={
            "name": "instrument-link",
            "type": "Element",
        }
    )
    group_link: List[str] = field(
        default_factory=list,
        metadata={
            "name": "group-link",
            "type": "Element",
        }
    )
    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
            "required": True,
        }
    )
    type: TypeValue = field(
        init=False,
        default=TypeValue.SIMPLE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    role: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    title: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    show: ShowValue = field(
        default=ShowValue.REPLACE,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )
    actuate: ActuateValue = field(
        default=ActuateValue.ON_REQUEST,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/1999/xlink",
        }
    )


@dataclass
class PartName:
    """The part-name type describes the name or abbreviation of a score-part
    element.

    Formatting attributes for the part-name element are deprecated in
    Version 2.0 in favor of the new part-name-display and part-
    abbreviation-display elements.
    """
    class Meta:
        name = "part-name"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PartSymbol:
    """The part-symbol type indicates how a symbol for a multi-staff part is
    indicated in the score; brace is the default value.

    The top-staff and bottom-staff attributes are used when the brace
    does not extend across the entire part. For example, in a 3-staff
    organ part, the top-staff will typically be 1 for the right hand,
    while the bottom-staff will typically be 2 for the left hand. Staff
    3 for the pedals is usually outside the brace. By default, the
    presence of a part-symbol element that does not extend across the
    entire part also indicates a corresponding change in the common
    barlines within a part.
    """
    class Meta:
        name = "part-symbol"

    value: Optional[GroupSymbolValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    top_staff: Optional[int] = field(
        default=None,
        metadata={
            "name": "top-staff",
            "type": "Attribute",
        }
    )
    bottom_staff: Optional[int] = field(
        default=None,
        metadata={
            "name": "bottom-staff",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Pedal:
    """The pedal type represents piano pedal marks, including damper and
    sostenuto pedal marks.

    The line attribute is yes if pedal lines are used. The sign attribute is yes if Ped, Sost, and * signs are used. For compatibility with older versions, the sign attribute is yes by default if the line attribute is no, and is no by default if the line attribute is yes. If the sign attribute is set to yes and the type is start or sostenuto, the abbreviated attribute is yes if the short P and S signs are used, and no if the full Ped and Sost signs are used. It is no by default. Otherwise the abbreviated attribute is ignored. The alignment attributes are ignored if the sign attribute is no.
    """
    class Meta:
        name = "pedal"

    type: Optional[PedalType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    sign: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    abbreviated: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PedalTuning:
    """
    The pedal-tuning type specifies the tuning of a single harp pedal.

    :ivar pedal_step: The pedal-step element defines the pitch step for
        a single harp pedal.
    :ivar pedal_alter: The pedal-alter element defines the chromatic
        alteration for a single harp pedal.
    """
    class Meta:
        name = "pedal-tuning"

    pedal_step: Optional[Step] = field(
        default=None,
        metadata={
            "name": "pedal-step",
            "type": "Element",
            "required": True,
        }
    )
    pedal_alter: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "pedal-alter",
            "type": "Element",
            "required": True,
        }
    )


@dataclass
class PerMinute:
    """The per-minute type can be a number, or a text description including
    numbers.

    If a font is specified, it overrides the font specified for the
    overall metronome element. This allows separate specification of a
    music font for the beat-unit and a text font for the numeric value,
    in cases where a single metronome font is not used.
    """
    class Meta:
        name = "per-minute"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )


@dataclass
class Pitch:
    """
    Pitch is represented as a combination of the step of the diatonic scale,
    the chromatic alteration, and the octave.
    """
    class Meta:
        name = "pitch"

    step: Optional[Step] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    alter: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    octave: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )


@dataclass
class Pitched:
    """The pitched-value type represents pictograms for pitched percussion
    instruments.

    The smufl attribute is used to distinguish different SMuFL glyphs
    for a particular pictogram within the Tuned mallet percussion
    pictograms range.
    """
    class Meta:
        name = "pitched"

    value: Optional[PitchedValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class PlacementText:
    """
    The placement-text type represents a text element with print-style and
    placement attribute groups.
    """
    class Meta:
        name = "placement-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Play:
    """The play type specifies playback techniques to be used in conjunction
    with the instrument-sound element.

    When used as part of a sound element, it applies to all notes going
    forward in score order. In multi-instrument parts, the affected
    instrument should be specified using the id attribute. When used as
    part of a note element, it applies to the current note only.

    :ivar ipa: The ipa element represents International Phonetic
        Alphabet (IPA) sounds for vocal music. String content is limited
        to IPA 2015 symbols represented in Unicode 13.0.
    :ivar mute:
    :ivar semi_pitched:
    :ivar other_play:
    :ivar id:
    """
    class Meta:
        name = "play"

    ipa: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    mute: List[Mute] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    semi_pitched: List[SemiPitched] = field(
        default_factory=list,
        metadata={
            "name": "semi-pitched",
            "type": "Element",
            "sequential": True,
        }
    )
    other_play: List[OtherPlay] = field(
        default_factory=list,
        metadata={
            "name": "other-play",
            "type": "Element",
            "sequential": True,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PrincipalVoice:
    """The principal-voice type represents principal and secondary voices in a
    score, either for analysis or for square bracket symbols that appear in a
    score.

    The element content is used for analysis and may be any text value.
    The symbol attribute indicates the type of symbol used. When used
    for analysis separate from any printed score markings, it should be
    set to none. Otherwise if the type is stop it should be set to
    plain.
    """
    class Meta:
        name = "principal-voice"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    symbol: Optional[PrincipalVoiceSymbol] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Release(Empty):
    """The release type indicates that a bend is a release rather than a normal
    bend or pre-bend.

    The offset attribute specifies where the release starts in terms of
    divisions relative to the current note. The first-beat and last-beat
    attributes of the parent bend element are relative to the original
    note position, not this offset value.
    """
    class Meta:
        name = "release"

    offset: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Repeat:
    """The repeat type represents repeat marks.

    The start of the repeat has a forward direction while the end of the
    repeat has a backward direction. The times and after-jump attributes
    are only used with backward repeats that are not part of an ending.
    The times attribute indicates the number of times the repeated
    section is played. The after-jump attribute indicates if the repeats
    are played after a jump due to a da capo or dal segno.
    """
    class Meta:
        name = "repeat"

    direction: Optional[BackwardForward] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    times: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    after_jump: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "after-jump",
            "type": "Attribute",
        }
    )
    winged: Optional[Winged] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Rest:
    """The rest element indicates notated rests or silences.

    Rest elements are usually empty, but placement on the staff can be
    specified using display-step and display-octave elements. If the
    measure attribute is set to yes, this indicates this is a complete
    measure rest.
    """
    class Meta:
        name = "rest"

    display_step: Optional[Step] = field(
        default=None,
        metadata={
            "name": "display-step",
            "type": "Element",
        }
    )
    display_octave: Optional[int] = field(
        default=None,
        metadata={
            "name": "display-octave",
            "type": "Element",
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )
    measure: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class RootStep:
    """The root-step type represents the pitch step of the root of the current
    chord within the harmony element.

    The text attribute indicates how the root should appear in a score
    if not using the element contents.
    """
    class Meta:
        name = "root-step"

    value: Optional[Step] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class ScoreInstrument:
    """The score-instrument type represents a single instrument within a score-
    part.

    As with the score-part type, each score-instrument has a required ID
    attribute, a name, and an optional abbreviation. A score-instrument
    type is also required if the score specifies MIDI 1.0 channels,
    banks, or programs. An initial midi-instrument assignment can also
    be made here. MusicXML software should be able to automatically
    assign reasonable channels and instruments without these elements in
    simple cases, such as where part names match General MIDI instrument
    names. The score-instrument element can also distinguish multiple
    instruments of the same type that are on the same part, such as
    Clarinet 1 and Clarinet 2 instruments within a Clarinets 1 and 2
    part.

    :ivar instrument_name: The instrument-name element is typically used
        within a software application, rather than appearing on the
        printed page of a score.
    :ivar instrument_abbreviation: The optional instrument-abbreviation
        element is typically used within a software application, rather
        than appearing on the printed page of a score.
    :ivar instrument_sound: The instrument-sound element describes the
        default timbre of the score-instrument. This description is
        independent of a particular virtual or MIDI instrument
        specification and allows playback to be shared more easily
        between applications and libraries.
    :ivar solo: The solo element is present if performance is intended
        by a solo instrument.
    :ivar ensemble: The ensemble element is present if performance is
        intended by an ensemble such as an orchestral section. The text
        of the ensemble element contains the size of the section, or is
        empty if the ensemble size is not specified.
    :ivar virtual_instrument:
    :ivar id:
    """
    class Meta:
        name = "score-instrument"

    instrument_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "instrument-name",
            "type": "Element",
            "required": True,
        }
    )
    instrument_abbreviation: Optional[str] = field(
        default=None,
        metadata={
            "name": "instrument-abbreviation",
            "type": "Element",
        }
    )
    instrument_sound: Optional[str] = field(
        default=None,
        metadata={
            "name": "instrument-sound",
            "type": "Element",
        }
    )
    solo: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    ensemble: Optional[Union[int, PositiveIntegerOrEmptyValue]] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    virtual_instrument: Optional[VirtualInstrument] = field(
        default=None,
        metadata={
            "name": "virtual-instrument",
            "type": "Element",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class Segno:
    """The segno type is the visual indicator of a segno sign.

    The exact glyph can be specified with the smufl attribute. A sound
    element is also needed to guide playback applications reliably.
    """
    class Meta:
        name = "segno"

    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"segno\c*",
        }
    )


@dataclass
class Slash:
    """The slash type is used to indicate that slash notation is to be used.

    If the slash is on every beat, use-stems is no (the default). To
    indicate rhythms but not pitches, use-stems is set to yes. The type
    attribute indicates whether this is the start or stop of a slash
    notation style. The use-dots attribute works as for the beat-repeat
    element, and only has effect if use-stems is no.

    :ivar slash_type: The slash-type element indicates the graphical
        note type to use for the display of repetition marks.
    :ivar slash_dot: The slash-dot element is used to specify any
        augmentation dots in the note type used to display repetition
        marks.
    :ivar except_voice: The except-voice element is used to specify a
        combination of slash notation and regular notation. Any note
        elements that are in voices specified by the except-voice
        elements are displayed in normal notation, in addition to the
        slash notation that is always displayed.
    :ivar type:
    :ivar use_dots:
    :ivar use_stems:
    """
    class Meta:
        name = "slash"

    slash_type: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "name": "slash-type",
            "type": "Element",
        }
    )
    slash_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "slash-dot",
            "type": "Element",
        }
    )
    except_voice: List[str] = field(
        default_factory=list,
        metadata={
            "name": "except-voice",
            "type": "Element",
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    use_dots: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "use-dots",
            "type": "Attribute",
        }
    )
    use_stems: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "use-stems",
            "type": "Attribute",
        }
    )


@dataclass
class Slide:
    """Glissando and slide types both indicate rapidly moving from one pitch to
    the other so that individual notes are not discerned.

    A slide is continuous between the two pitches and defaults to a
    solid line. The optional text for a is printed alongside the line.
    """
    class Meta:
        name = "slide"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    accelerate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    beats: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("2"),
        }
    )
    first_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "first-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    last_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "last-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Slur:
    """Slur types are empty.

    Most slurs are represented with two elements: one with a start type,
    and one with a stop type. Slurs can add more elements using a
    continue type. This is typically used to specify the formatting of
    cross-system slurs, or to specify the shape of very complex slurs.
    """
    class Meta:
        name = "slur"

    type: Optional[StartStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: int = field(
        default=1,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    orientation: Optional[OverUnder] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    bezier_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-x",
            "type": "Attribute",
        }
    )
    bezier_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-y",
            "type": "Attribute",
        }
    )
    bezier_x2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-x2",
            "type": "Attribute",
        }
    )
    bezier_y2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-y2",
            "type": "Attribute",
        }
    )
    bezier_offset: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-offset",
            "type": "Attribute",
        }
    )
    bezier_offset2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-offset2",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StaffDivide:
    """
    The staff-divide element represents the staff division arrow symbols found
    at SMuFL code points U+E00B, U+E00C, and U+E00D.
    """
    class Meta:
        name = "staff-divide"

    type: Optional[StaffDivideSymbol] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StaffTuning:
    """
    The staff-tuning type specifies the open, non-capo tuning of the lines on a
    tablature staff.

    :ivar tuning_step: The tuning-step element is represented like the
        step element, with a different name to reflect its different
        function in string tuning.
    :ivar tuning_alter: The tuning-alter element is represented like the
        alter element, with a different name to reflect its different
        function in string tuning.
    :ivar tuning_octave: The tuning-octave element is represented like
        the octave element, with a different name to reflect its
        different function in string tuning.
    :ivar line:
    """
    class Meta:
        name = "staff-tuning"

    tuning_step: Optional[Step] = field(
        default=None,
        metadata={
            "name": "tuning-step",
            "type": "Element",
            "required": True,
        }
    )
    tuning_alter: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "tuning-alter",
            "type": "Element",
        }
    )
    tuning_octave: Optional[int] = field(
        default=None,
        metadata={
            "name": "tuning-octave",
            "type": "Element",
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )
    line: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class Stem:
    """Stems can be down, up, none, or double.

    For down and up stems, the position attributes can be used to
    specify stem length. The relative values specify the end of the stem
    relative to the program default. Default values specify an absolute
    end stem position. Negative values of relative-y that would flip a
    stem instead of shortening it are ignored. A stem element associated
    with a rest refers to a stemlet.
    """
    class Meta:
        name = "stem"

    value: Optional[StemValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Stick:
    """The stick type represents pictograms where the material of the stick,
    mallet, or beater is included.The parentheses and dashed-circle attributes
    indicate the presence of these marks around the round beater part of a
    pictogram.

    Values for these attributes are "no" if not present.
    """
    class Meta:
        name = "stick"

    stick_type: Optional[StickType] = field(
        default=None,
        metadata={
            "name": "stick-type",
            "type": "Element",
            "required": True,
        }
    )
    stick_material: Optional[StickMaterial] = field(
        default=None,
        metadata={
            "name": "stick-material",
            "type": "Element",
            "required": True,
        }
    )
    tip: Optional[TipDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    dashed_circle: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "dashed-circle",
            "type": "Attribute",
        }
    )


@dataclass
class String:
    """The string type is used with tablature notation, regular notation (where
    it is often circled), and chord diagrams.

    String numbers start with 1 for the highest pitched full-length
    string.
    """
    class Meta:
        name = "string"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StringMute:
    """
    The string-mute type represents string mute on and mute off symbols.
    """
    class Meta:
        name = "string-mute"

    type: Optional[OnOff] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StyleText:
    """
    The style-text type represents a text element with a print-style attribute
    group.
    """
    class Meta:
        name = "style-text"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Supports:
    """The supports type indicates if a MusicXML encoding supports a particular
    MusicXML element.

    This is recommended for elements like beam, stem, and accidental,
    where the absence of an element is ambiguous if you do not know if
    the encoding supports that element. For Version 2.0, the supports
    element is expanded to allow programs to indicate support for
    particular attributes or particular values. This lets applications
    communicate, for example, that all system and/or page breaks are
    contained in the MusicXML file.
    """
    class Meta:
        name = "supports"

    type: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    element: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    attribute: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    value: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Swing:
    """The swing element specifies whether or not to use swing playback, where
    consecutive on-beat / off-beat eighth or 16th notes are played with unequal
    nominal durations.

    The straight element specifies that no swing is present, so
    consecutive notes have equal durations. The first and second
    elements are positive integers that specify the ratio between
    durations of consecutive notes. For example, a first element with a
    value of 2 and a second element with a value of 1 applied to eighth
    notes specifies a quarter note / eighth note tuplet playback, where
    the first note is twice as long as the second note. Ratios should be
    specified with the smallest integers possible. For example, a ratio
    of 6 to 4 should be specified as 3 to 2 instead. The optional swing-
    type element specifies the note type, either eighth or 16th, to
    which the ratio is applied. The value is eighth if this element is
    not present. The optional swing-style element is a string describing
    the style of swing used. The swing element has no effect for
    playback of grace notes, notes where a type element is not present,
    and notes where the specified duration is different than the nominal
    value associated with the specified type. If a swung note has attack
    and release attributes, those values modify the swung playback.
    """
    class Meta:
        name = "swing"

    straight: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    first: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    second: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    swing_type: Optional[SwingTypeValue] = field(
        default=None,
        metadata={
            "name": "swing-type",
            "type": "Element",
        }
    )
    swing_style: Optional[str] = field(
        default=None,
        metadata={
            "name": "swing-style",
            "type": "Element",
        }
    )


@dataclass
class Sync:
    """The sync type specifies the style that a score following application
    should use the synchronize an accompaniment with a performer.

    If this type is not included in a score, default synchronization
    depends on the application. The optional latency attribute specifies
    a time in milliseconds that the listening application should expect
    from the performer. The optional player and time-only attributes
    restrict the element to apply to a single player or set of times
    through a repeated section, respectively.
    """
    class Meta:
        name = "sync"

    type: Optional[SyncType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    latency: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    player: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )


@dataclass
class Tap:
    """The tap type indicates a tap on the fretboard.

    The text content allows specification of the notation; + and T are
    common choices. If the element is empty, the hand attribute is used
    to specify the symbol to use. The hand attribute is ignored if the
    tap glyph is already specified by the text content. If neither text
    content nor the hand attribute are present, the display is
    application-specific.
    """
    class Meta:
        name = "tap"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    hand: Optional[TapHand] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class TextElementData:
    """The text-element-data type represents a syllable or portion of a
    syllable for lyric text underlay.

    A hyphen in the string content should only be used for an actual
    hyphenated word. Language names for text elements come from ISO 639,
    with optional country subcodes from ISO 3166.
    """
    class Meta:
        name = "text-element-data"

    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    underline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    overline: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    line_through: Optional[int] = field(
        default=None,
        metadata={
            "name": "line-through",
            "type": "Attribute",
            "min_inclusive": 0,
            "max_inclusive": 3,
        }
    )
    rotation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    letter_spacing: Optional[Union[Decimal, NumberOrNormalValue]] = field(
        default=None,
        metadata={
            "name": "letter-spacing",
            "type": "Attribute",
        }
    )
    lang: Optional[Union[str, LangValue]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "namespace": "http://www.w3.org/XML/1998/namespace",
        }
    )
    dir: Optional[TextDirection] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Tie:
    """The tie element indicates that a tie begins or ends with this note.

    If the tie element applies only particular times through a repeat,
    the time-only attribute indicates which times to apply it. The tie
    element indicates sound; the tied element indicates notation.
    """
    class Meta:
        name = "tie"

    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )


@dataclass
class Tied:
    """The tied element represents the notated tie.

    The tie element represents the tie sound. The number attribute is
    rarely needed to disambiguate ties, since note pitches will usually
    suffice. The attribute is implied rather than defaulting to 1 as
    with most elements. It is available for use in more complex tied
    notation situations. Ties that join two notes of the same pitch
    together should be represented with a tied element on the first note
    with type="start" and a tied element on the second note with
    type="stop".  This can also be done if the two notes being tied are
    enharmonically equivalent, but have different step values. It is not
    recommended to use tied elements to join two notes with
    enharmonically inequivalent pitches. Ties that indicate that an
    instrument should be undamped are specified with a single tied
    element with type="let-ring". Ties that are visually attached to
    only one note, other than undamped ties, should be specified with
    two tied elements on the same note, first type="start" then
    type="stop". This can be used to represent ties into or out of
    repeated sections or codas.
    """
    class Meta:
        name = "tied"

    type: Optional[TiedType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    orientation: Optional[OverUnder] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    bezier_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-x",
            "type": "Attribute",
        }
    )
    bezier_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-y",
            "type": "Attribute",
        }
    )
    bezier_x2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-x2",
            "type": "Attribute",
        }
    )
    bezier_y2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-y2",
            "type": "Attribute",
        }
    )
    bezier_offset: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-offset",
            "type": "Attribute",
        }
    )
    bezier_offset2: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bezier-offset2",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class TimeModification:
    """Time modification indicates tuplets, double-note tremolos, and other
    durational changes.

    A time-modification element shows how the cumulative, sounding
    effect of tuplets and double-note tremolos compare to the written
    note type represented by the type and dot elements. Nested tuplets
    and other notations that use more detailed information need both the
    time-modification and tuplet elements to be represented accurately.

    :ivar actual_notes: The actual-notes element describes how many
        notes are played in the time usually occupied by the number in
        the normal-notes element.
    :ivar normal_notes: The normal-notes element describes how many
        notes are usually played in the time occupied by the number in
        the actual-notes element.
    :ivar normal_type: If the type associated with the number in the
        normal-notes element is different than the current note type
        (e.g., a quarter note within an eighth note triplet), then the
        normal-notes type (e.g. eighth) is specified in the normal-type
        and normal-dot elements.
    :ivar normal_dot: The normal-dot element is used to specify dotted
        normal tuplet types.
    """
    class Meta:
        name = "time-modification"

    actual_notes: Optional[int] = field(
        default=None,
        metadata={
            "name": "actual-notes",
            "type": "Element",
            "required": True,
        }
    )
    normal_notes: Optional[int] = field(
        default=None,
        metadata={
            "name": "normal-notes",
            "type": "Element",
            "required": True,
        }
    )
    normal_type: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "name": "normal-type",
            "type": "Element",
        }
    )
    normal_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "normal-dot",
            "type": "Element",
        }
    )


@dataclass
class Tremolo:
    """The tremolo ornament can be used to indicate single-note, double-note,
    or unmeasured tremolos.

    Single-note tremolos use the single type, double-note tremolos use
    the start and stop types, and unmeasured tremolos use the unmeasured
    type. The default is "single" for compatibility with Version 1.1.
    The text of the element indicates the number of tremolo marks and is
    an integer from 0 to 8. Note that the number of attached beams is
    not included in this value, but is represented separately using the
    beam element. The value should be 0 for unmeasured tremolos. When
    using double-note tremolos, the duration of each note in the tremolo
    should correspond to half of the notated type value. A time-
    modification element should also be added with an actual-notes value
    of 2 and a normal-notes value of 1. If used within a tuplet, this
    2/1 ratio should be multiplied by the existing tuplet ratio. The
    smufl attribute specifies the glyph to use from the SMuFL Tremolos
    range for an unmeasured tremolo. It is ignored for other tremolo
    types. The SMuFL buzzRoll glyph is used by default if the attribute
    is missing. Using repeater beams for indicating tremolos is
    deprecated as of MusicXML 3.0.
    """
    class Meta:
        name = "tremolo"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
            "min_inclusive": 0,
            "max_inclusive": 8,
        }
    )
    type: TremoloType = field(
        default=TremoloType.SINGLE,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class TupletDot:
    """
    The tuplet-dot type is used to specify dotted tuplet types.
    """
    class Meta:
        name = "tuplet-dot"

    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class TupletNumber:
    """
    The tuplet-number type indicates the number of notes for this portion of
    the tuplet.
    """
    class Meta:
        name = "tuplet-number"

    value: Optional[int] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class TupletType:
    """
    The tuplet-type type indicates the graphical note type of the notes for
    this portion of the tuplet.
    """
    class Meta:
        name = "tuplet-type"

    value: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )


@dataclass
class Unpitched:
    """The unpitched type represents musical elements that are notated on the
    staff but lack definite pitch, such as unpitched percussion and speaking
    voice.

    If the child elements are not present, the note is placed on the
    middle line of the staff. This is generally used with a one-line
    staff. Notes in percussion clef should always use an unpitched
    element rather than a pitch element.
    """
    class Meta:
        name = "unpitched"

    display_step: Optional[Step] = field(
        default=None,
        metadata={
            "name": "display-step",
            "type": "Element",
        }
    )
    display_octave: Optional[int] = field(
        default=None,
        metadata={
            "name": "display-octave",
            "type": "Element",
            "min_inclusive": 0,
            "max_inclusive": 9,
        }
    )


@dataclass
class WavyLine:
    """Wavy lines are one way to indicate trills and vibrato.

    When used with a barline element, they should always have
    type="continue" set. The smufl attribute specifies a particular wavy
    line glyph from the SMuFL Multi-segment lines range.
    """
    class Meta:
        name = "wavy-line"

    type: Optional[StartStopContinue] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"(wiggle\c+)|(guitar\c*VibratoStroke)",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    start_note: Optional[StartNote] = field(
        default=None,
        metadata={
            "name": "start-note",
            "type": "Attribute",
        }
    )
    trill_step: Optional[TrillStep] = field(
        default=None,
        metadata={
            "name": "trill-step",
            "type": "Attribute",
        }
    )
    two_note_turn: Optional[TwoNoteTurn] = field(
        default=None,
        metadata={
            "name": "two-note-turn",
            "type": "Attribute",
        }
    )
    accelerate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    beats: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("2"),
        }
    )
    second_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "second-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    last_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "last-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )


@dataclass
class Wedge:
    """The wedge type represents crescendo and diminuendo wedge symbols.

    The type attribute is crescendo for the start of a wedge that is
    closed at the left side, and diminuendo for the start of a wedge
    that is closed on the right side. Spread values are measured in
    tenths; those at the start of a crescendo wedge or end of a
    diminuendo wedge are ignored. The niente attribute is yes if a
    circle appears at the point of the wedge, indicating a crescendo
    from nothing or diminuendo to nothing. It is no by default, and used
    only when the type is crescendo, or the type is stop for a wedge
    that began with a diminuendo type. The line-type is solid if not
    specified.
    """
    class Meta:
        name = "wedge"

    type: Optional[WedgeType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    spread: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    niente: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    line_type: Optional[LineType] = field(
        default=None,
        metadata={
            "name": "line-type",
            "type": "Attribute",
        }
    )
    dash_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "dash-length",
            "type": "Attribute",
        }
    )
    space_length: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "space-length",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Wood:
    """The wood type represents pictograms for wood percussion instruments.

    The smufl attribute is used to distinguish different SMuFL stylistic
    alternates.
    """
    class Meta:
        name = "wood"

    value: Optional[WoodValue] = field(
        default=None,
        metadata={
            "required": True,
        }
    )
    smufl: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"pict\c+",
        }
    )


@dataclass
class Appearance:
    """The appearance type controls general graphical settings for the music's
    final form appearance on a printed page of display.

    This includes support for line widths, definitions for note sizes,
    and standard distances between notation elements, plus an extension
    element for other aspects of appearance.
    """
    class Meta:
        name = "appearance"

    line_width: List[LineWidth] = field(
        default_factory=list,
        metadata={
            "name": "line-width",
            "type": "Element",
        }
    )
    note_size: List[NoteSize] = field(
        default_factory=list,
        metadata={
            "name": "note-size",
            "type": "Element",
        }
    )
    distance: List[Distance] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    glyph: List[Glyph] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    other_appearance: List[OtherAppearance] = field(
        default_factory=list,
        metadata={
            "name": "other-appearance",
            "type": "Element",
        }
    )


@dataclass
class Backup:
    """The backup and forward elements are required to coordinate multiple
    voices in one part, including music on multiple staves.

    The backup type is generally used to move between voices and staves.
    Thus the backup element does not include voice or staff elements.
    Duration values should always be positive, and should not cross
    measure boundaries or mid-measure changes in the divisions value.

    :ivar duration: Duration is a positive number specified in division
        units. This is the intended duration vs. notated duration (for
        instance, differences in dotted notes in Baroque-era music).
        Differences in duration specific to an interpretation or
        performance should be represented using the note element's
        attack and release attributes. The duration element moves the
        musical position when used in backup elements, forward elements,
        and note elements that do not contain a chord child element.
    :ivar footnote:
    :ivar level:
    """
    class Meta:
        name = "backup"

    duration: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
            "min_exclusive": Decimal("0"),
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Barline:
    """If a barline is other than a normal single barline, it should be
    represented by a barline type that describes it.

    This includes information about repeats and multiple endings, as well as line style. Barline data is on the same level as the other musical data in a score - a child of a measure in a partwise score, or a part in a timewise score. This allows for barlines within measures, as in dotted barlines that subdivide measures in complex meters. The two fermata elements allow for fermatas on both sides of the barline (the lower one inverted).
    Barlines have a location attribute to make it easier to process barlines independently of the other musical data in a score. It is often easier to set up measures separately from entering notes. The location attribute must match where the barline element occurs within the rest of the musical data in the score. If location is left, it should be the first element in the measure, aside from the print, bookmark, and link elements. If location is right, it should be the last element, again with the possible exception of the print, bookmark, and link elements. If no location is specified, the right barline is the default. The segno, coda, and divisions attributes work the same way as in the sound element. They are used for playback when barline elements contain segno or coda child elements.
    """
    class Meta:
        name = "barline"

    bar_style: Optional[BarStyleColor] = field(
        default=None,
        metadata={
            "name": "bar-style",
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    wavy_line: Optional[WavyLine] = field(
        default=None,
        metadata={
            "name": "wavy-line",
            "type": "Element",
        }
    )
    segno: Optional[Segno] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    coda: Optional[Coda] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    fermata: List[Fermata] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 2,
        }
    )
    ending: Optional[Ending] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    repeat: Optional[Repeat] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    location: RightLeftMiddle = field(
        default=RightLeftMiddle.RIGHT,
        metadata={
            "type": "Attribute",
        }
    )
    segno_attribute: Optional[str] = field(
        default=None,
        metadata={
            "name": "segno",
            "type": "Attribute",
        }
    )
    coda_attribute: Optional[str] = field(
        default=None,
        metadata={
            "name": "coda",
            "type": "Attribute",
        }
    )
    divisions: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Bass:
    """The bass type is used to indicate a bass note in popular music chord
    symbols, e.g. G/C.

    It is generally not used in functional harmony, as inversion is
    generally not used in pop chord symbols. As with root, it is divided
    into step and alter elements, similar to pitches. The arrangement
    attribute specifies where the bass is displayed relative to what
    precedes it.

    :ivar bass_separator: The optional bass-separator element indicates
        that text, rather than a line or slash, separates the bass from
        what precedes it.
    :ivar bass_step:
    :ivar bass_alter: The bass-alter element represents the chromatic
        alteration of the bass of the current chord within the harmony
        element. In some chord styles, the text for the bass-step
        element may include bass-alter information. In that case, the
        print-object attribute of the bass-alter element can be set to
        no. The location attribute indicates whether the alteration
        should appear to the left or the right of the bass-step; it is
        right if not specified.
    :ivar arrangement:
    """
    class Meta:
        name = "bass"

    bass_separator: Optional[StyleText] = field(
        default=None,
        metadata={
            "name": "bass-separator",
            "type": "Element",
        }
    )
    bass_step: Optional[BassStep] = field(
        default=None,
        metadata={
            "name": "bass-step",
            "type": "Element",
            "required": True,
        }
    )
    bass_alter: Optional[HarmonyAlter] = field(
        default=None,
        metadata={
            "name": "bass-alter",
            "type": "Element",
        }
    )
    arrangement: Optional[HarmonyArrangement] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Bend:
    """The bend type is used in guitar notation and tablature.

    A single note with a bend and release will contain two bend
    elements: the first to represent the bend and the second to
    represent the release. The shape attribute distinguishes between the
    angled bend symbols commonly used in standard notation and the
    curved bend symbols commonly used in both tablature and standard
    notation.

    :ivar bend_alter: The bend-alter element indicates the number of
        semitones in the bend, similar to the alter element. As with the
        alter element, numbers like 0.5 can be used to indicate
        microtones. Negative values indicate pre-bends or releases. The
        pre-bend and release elements are used to distinguish what is
        intended. Because the bend-alter element represents the number
        of steps in the bend, a release after a bend has a negative
        bend-alter value, not a zero value.
    :ivar pre_bend: The pre-bend element indicates that a bend is a pre-
        bend rather than a normal bend or a release.
    :ivar release:
    :ivar with_bar: The with-bar element indicates that the bend is to
        be done at the bridge with a whammy or vibrato bar. The content
        of the element indicates how this should be notated. Content
        values of "scoop" and "dip" refer to the SMuFL
        guitarVibratoBarScoop and guitarVibratoBarDip glyphs.
    :ivar shape:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar accelerate:
    :ivar beats:
    :ivar first_beat:
    :ivar last_beat:
    """
    class Meta:
        name = "bend"

    bend_alter: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "bend-alter",
            "type": "Element",
            "required": True,
        }
    )
    pre_bend: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "pre-bend",
            "type": "Element",
        }
    )
    release: Optional[Release] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    with_bar: Optional[PlacementText] = field(
        default=None,
        metadata={
            "name": "with-bar",
            "type": "Element",
        }
    )
    shape: Optional[BendShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    accelerate: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    beats: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("2"),
        }
    )
    first_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "first-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )
    last_beat: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "last-beat",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
            "max_inclusive": Decimal("100"),
        }
    )


@dataclass
class Credit:
    """The credit type represents the appearance of the title, composer,
    arranger, lyricist, copyright, dedication, and other text, symbols, and
    graphics that commonly appear on the first page of a score.

    The credit-words, credit-symbol, and credit-image elements are
    similar to the words, symbol, and image elements for directions.
    However, since the credit is not part of a measure, the default-x
    and default-y attributes adjust the origin relative to the bottom
    left-hand corner of the page. The enclosure for credit-words and
    credit-symbol is none by default. By default, a series of credit-
    words and credit-symbol elements within a single credit element
    follow one another in sequence visually. Non-positional formatting
    attributes are carried over from the previous element by default.
    The page attribute for the credit element specifies the page number
    where the credit should appear. This is an integer value that starts
    with 1 for the first page. Its value is 1 by default. Since credits
    occur before the music, these page numbers do not refer to the page
    numbering specified by the print element's page-number attribute.
    The credit-type element indicates the purpose behind a credit.
    Multiple types of data may be combined in a single credit, so
    multiple elements may be used. Standard values include page number,
    title, subtitle, composer, arranger, lyricist, rights, and part
    name.
    """
    class Meta:
        name = "credit"

    credit_type: List[str] = field(
        default_factory=list,
        metadata={
            "name": "credit-type",
            "type": "Element",
        }
    )
    link: List[Link] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    bookmark: List[Bookmark] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    credit_image: Optional[Image] = field(
        default=None,
        metadata={
            "name": "credit-image",
            "type": "Element",
        }
    )
    credit_words: List[FormattedTextId] = field(
        default_factory=list,
        metadata={
            "name": "credit-words",
            "type": "Element",
            "sequential": True,
        }
    )
    credit_symbol: List[FormattedSymbolId] = field(
        default_factory=list,
        metadata={
            "name": "credit-symbol",
            "type": "Element",
            "sequential": True,
        }
    )
    page: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Degree:
    """The degree type is used to add, alter, or subtract individual notes in
    the chord.

    The print-object attribute can be used to keep the degree from
    printing separately when it has already taken into account in the
    text attribute of the kind element. The degree-value and degree-type
    text attributes specify how the value and type of the degree should
    be displayed. A harmony of kind "other" can be spelled explicitly by
    using a series of degree elements together with a root.
    """
    class Meta:
        name = "degree"

    degree_value: Optional[DegreeValue] = field(
        default=None,
        metadata={
            "name": "degree-value",
            "type": "Element",
            "required": True,
        }
    )
    degree_alter: Optional[DegreeAlter] = field(
        default=None,
        metadata={
            "name": "degree-alter",
            "type": "Element",
            "required": True,
        }
    )
    degree_type: Optional[DegreeType] = field(
        default=None,
        metadata={
            "name": "degree-type",
            "type": "Element",
            "required": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )


@dataclass
class Encoding:
    """The encoding element contains information about who did the digital
    encoding, when, with what software, and in what aspects.

    Standard type values for the encoder element are music, words, and
    arrangement, but other types may be used. The type attribute is only
    needed when there are multiple encoder elements.
    """
    class Meta:
        name = "encoding"

    encoding_date: List[str] = field(
        default_factory=list,
        metadata={
            "name": "encoding-date",
            "type": "Element",
            "pattern": r"[^:Z]*",
        }
    )
    encoder: List[TypedText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    software: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    encoding_description: List[str] = field(
        default_factory=list,
        metadata={
            "name": "encoding-description",
            "type": "Element",
        }
    )
    supports: List[Supports] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Figure:
    """
    The figure type represents a single figure within a figured-bass element.

    :ivar prefix: Values for the prefix element include plus and the
        accidental values sharp, flat, natural, double-sharp, flat-flat,
        and sharp-sharp. The prefix element may contain additional
        values for symbols specific to particular figured bass styles.
    :ivar figure_number: A figure-number is a number. Overstrikes of the
        figure number are represented in the suffix element.
    :ivar suffix: Values for the suffix element include plus and the
        accidental values sharp, flat, natural, double-sharp, flat-flat,
        and sharp-sharp. Suffixes include both symbols that come after
        the figure number and those that overstrike the figure number.
        The suffix values slash, back-slash, and vertical are used for
        slashed numbers indicating chromatic alteration. The orientation
        and display of the slash usually depends on the figure number.
        The suffix element may contain additional values for symbols
        specific to particular figured bass styles.
    :ivar extend:
    :ivar footnote:
    :ivar level:
    """
    class Meta:
        name = "figure"

    prefix: Optional[StyleText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    figure_number: Optional[StyleText] = field(
        default=None,
        metadata={
            "name": "figure-number",
            "type": "Element",
        }
    )
    suffix: Optional[StyleText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    extend: Optional[Extend] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Forward:
    """The backup and forward elements are required to coordinate multiple
    voices in one part, including music on multiple staves.

    The forward element is generally used within voices and staves.
    Duration values should always be positive, and should not cross
    measure boundaries or mid-measure changes in the divisions value.

    :ivar duration: Duration is a positive number specified in division
        units. This is the intended duration vs. notated duration (for
        instance, differences in dotted notes in Baroque-era music).
        Differences in duration specific to an interpretation or
        performance should be represented using the note element's
        attack and release attributes. The duration element moves the
        musical position when used in backup elements, forward elements,
        and note elements that do not contain a chord child element.
    :ivar footnote:
    :ivar level:
    :ivar voice:
    :ivar staff: Staff assignment is only needed for music notated on
        multiple staves. Used by both notes and directions. Staff values
        are numbers, with 1 referring to the top-most staff in a part.
    """
    class Meta:
        name = "forward"

    duration: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
            "min_exclusive": Decimal("0"),
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    voice: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    staff: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class FrameNote:
    """The frame-note type represents each note included in the frame.

    An open string will have a fret value of 0, while a muted string
    will not be associated with a frame-note element.
    """
    class Meta:
        name = "frame-note"

    string: Optional[String] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    fret: Optional[Fret] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    fingering: Optional[Fingering] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    barre: Optional[Barre] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class HarmonMute:
    """
    The harmon-mute type represents the symbols used for harmon mutes in brass
    notation.
    """
    class Meta:
        name = "harmon-mute"

    harmon_closed: Optional[HarmonClosed] = field(
        default=None,
        metadata={
            "name": "harmon-closed",
            "type": "Element",
            "required": True,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HarpPedals:
    """The harp-pedals type is used to create harp pedal diagrams.

    The pedal-step and pedal-alter elements use the same values as the
    step and alter elements. For easiest reading, the pedal-tuning
    elements should follow standard harp pedal order, with pedal-step
    values of D, C, B, E, F, G, and A.
    """
    class Meta:
        name = "harp-pedals"

    pedal_tuning: List[PedalTuning] = field(
        default_factory=list,
        metadata={
            "name": "pedal-tuning",
            "type": "Element",
            "min_occurs": 1,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class HeelToe(EmptyPlacement):
    """The heel and toe elements are used with organ pedals.

    The substitution value is "no" if the attribute is not present.
    """
    class Meta:
        name = "heel-toe"

    substitution: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Hole:
    """
    The hole type represents the symbols used for woodwind and brass fingerings
    as well as other notations.

    :ivar hole_type: The content of the optional hole-type element
        indicates what the hole symbol represents in terms of instrument
        fingering or other techniques.
    :ivar hole_closed:
    :ivar hole_shape: The optional hole-shape element indicates the
        shape of the hole symbol; the default is a circle.
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar placement:
    """
    class Meta:
        name = "hole"

    hole_type: Optional[str] = field(
        default=None,
        metadata={
            "name": "hole-type",
            "type": "Element",
        }
    )
    hole_closed: Optional[HoleClosed] = field(
        default=None,
        metadata={
            "name": "hole-closed",
            "type": "Element",
            "required": True,
        }
    )
    hole_shape: Optional[str] = field(
        default=None,
        metadata={
            "name": "hole-shape",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Key:
    """The key type represents a key signature.

    Both traditional and non-traditional key signatures are supported.
    The optional number attribute refers to staff numbers. If absent,
    the key signature applies to all staves in the part. Key signatures
    appear at the start of each system unless the print-object attribute
    has been set to "no".

    :ivar cancel:
    :ivar fifths:
    :ivar mode:
    :ivar key_step: Non-traditional key signatures are represented using
        a list of altered tones. The key-step element indicates the
        pitch step to be altered, represented using the same names as in
        the step element.
    :ivar key_alter: Non-traditional key signatures are represented
        using a list of altered tones. The key-alter element represents
        the alteration for a given pitch step, represented with
        semitones in the same manner as the alter element.
    :ivar key_accidental: Non-traditional key signatures are represented
        using a list of altered tones. The key-accidental element
        indicates the accidental to be displayed in the key signature,
        represented in the same manner as the accidental element. It is
        used for disambiguating microtonal accidentals.
    :ivar key_octave: The optional list of key-octave elements is used
        to specify in which octave each element of the key signature
        appears.
    :ivar number:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar print_object:
    :ivar id:
    """
    class Meta:
        name = "key"

    cancel: Optional[Cancel] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    fifths: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    mode: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    key_step: List[Step] = field(
        default_factory=list,
        metadata={
            "name": "key-step",
            "type": "Element",
            "sequential": True,
        }
    )
    key_alter: List[Decimal] = field(
        default_factory=list,
        metadata={
            "name": "key-alter",
            "type": "Element",
            "sequential": True,
        }
    )
    key_accidental: List[KeyAccidental] = field(
        default_factory=list,
        metadata={
            "name": "key-accidental",
            "type": "Element",
            "sequential": True,
        }
    )
    key_octave: List[KeyOctave] = field(
        default_factory=list,
        metadata={
            "name": "key-octave",
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Listen:
    """The listen and listening types, new in Version 4.0, specify different
    ways that a score following or machine listening application can interact
    with a performer.

    The listen type handles interactions that are specific to a note. If
    multiple child elements of the same type are present, they should
    have distinct player and/or time-only attributes.
    """
    class Meta:
        name = "listen"

    assess: List[Assess] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    wait: List[Wait] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    other_listen: List[OtherListening] = field(
        default_factory=list,
        metadata={
            "name": "other-listen",
            "type": "Element",
        }
    )


@dataclass
class Listening:
    """The listen and listening types, new in Version 4.0, specify different
    ways that a score following or machine listening application can interact
    with a performer.

    The listening type handles interactions that change the state of the
    listening application from the specified point in the performance
    onward. If multiple child elements of the same type are present,
    they should have distinct player and/or time-only attributes. The
    offset element is used to indicate that the listening change takes
    place offset from the current score position. If the listening
    element is a child of a direction element, the listening offset
    element overrides the direction offset element if both elements are
    present. Note that the offset reflects the intended musical position
    for the change in state. It should not be used to compensate for
    latency issues in particular hardware configurations.
    """
    class Meta:
        name = "listening"

    sync: List[Sync] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    other_listening: List[OtherListening] = field(
        default_factory=list,
        metadata={
            "name": "other-listening",
            "type": "Element",
            "sequential": True,
        }
    )
    offset: Optional[Offset] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Lyric:
    """The lyric type represents text underlays for lyrics.

    Two text elements that are not separated by an elision element are
    part of the same syllable, but may have different text formatting.
    The MusicXML XSD is more strict than the DTD in enforcing this by
    disallowing a second syllabic element unless preceded by an elision
    element. The lyric number indicates multiple lines, though a name
    can be used as well. Common name examples are verse and chorus.
    Justification is center by default; placement is below by default.
    Vertical alignment is to the baseline of the text and horizontal
    alignment matches justification. The print-object attribute can
    override a note's print-lyric attribute in cases where only some
    lyrics on a note are printed, as when lyrics for later verses are
    printed in a block of text rather than with each note. The time-only
    attribute precisely specifies which lyrics are to be sung which time
    through a repeated section.

    :ivar syllabic:
    :ivar text:
    :ivar elision:
    :ivar extend:
    :ivar laughing: The laughing element represents a laughing voice.
    :ivar humming: The humming element represents a humming voice.
    :ivar end_line: The end-line element comes from RP-017 for Standard
        MIDI File Lyric meta-events. It facilitates lyric display for
        Karaoke and similar applications.
    :ivar end_paragraph: The end-paragraph element comes from RP-017 for
        Standard MIDI File Lyric meta-events. It facilitates lyric
        display for Karaoke and similar applications.
    :ivar footnote:
    :ivar level:
    :ivar number:
    :ivar name:
    :ivar justify:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar placement:
    :ivar color:
    :ivar print_object:
    :ivar time_only:
    :ivar id:
    """
    class Meta:
        name = "lyric"

    syllabic: List[Syllabic] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    text: List[TextElementData] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    elision: List[Elision] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    extend: List[Extend] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 2,
            "sequential": True,
        }
    )
    laughing: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    humming: Optional[Empty] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    end_line: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "end-line",
            "type": "Element",
        }
    )
    end_paragraph: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "end-paragraph",
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    number: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class MeasureStyle:
    """A measure-style indicates a special way to print partial to multiple
    measures within a part.

    This includes multiple rests over several measures, repeats of
    beats, single, or multiple measures, and use of slash notation. The
    multiple-rest and measure-repeat elements indicate the number of
    measures covered in the element content. The beat-repeat and slash
    elements can cover partial measures. All but the multiple-rest
    element use a type attribute to indicate starting and stopping the
    use of the style. The optional number attribute specifies the staff
    number from top to bottom on the system, as with clef.
    """
    class Meta:
        name = "measure-style"

    multiple_rest: Optional[MultipleRest] = field(
        default=None,
        metadata={
            "name": "multiple-rest",
            "type": "Element",
        }
    )
    measure_repeat: Optional[MeasureRepeat] = field(
        default=None,
        metadata={
            "name": "measure-repeat",
            "type": "Element",
        }
    )
    beat_repeat: Optional[BeatRepeat] = field(
        default=None,
        metadata={
            "name": "beat-repeat",
            "type": "Element",
        }
    )
    slash: Optional[Slash] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class MetronomeTuplet(TimeModification):
    """
    The metronome-tuplet type uses the same element structure as the time-
    modification element along with some attributes from the tuplet element.
    """
    class Meta:
        name = "metronome-tuplet"

    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    bracket: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    show_number: Optional[ShowTuplet] = field(
        default=None,
        metadata={
            "name": "show-number",
            "type": "Attribute",
        }
    )


@dataclass
class Mordent(EmptyTrillSound):
    """The mordent type is used for both represents the mordent sign with the
    vertical line and the inverted-mordent sign without the line.

    The long attribute is "no" by default. The approach and departure
    attributes are used for compound ornaments, indicating how the
    beginning and ending of the ornament look relative to the main part
    of the mordent.
    """
    class Meta:
        name = "mordent"

    long: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    approach: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    departure: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class NameDisplay:
    """The name-display type is used for exact formatting of multi-font text in
    part and group names to the left of the system.

    The print-object attribute can be used to determine what, if
    anything, is printed at the start of each system. Enclosure for the
    display-text element is none by default. Language for the display-
    text element is Italian ("it") by default.
    """
    class Meta:
        name = "name-display"

    display_text: List[FormattedText] = field(
        default_factory=list,
        metadata={
            "name": "display-text",
            "type": "Element",
            "sequential": True,
        }
    )
    accidental_text: List[AccidentalText] = field(
        default_factory=list,
        metadata={
            "name": "accidental-text",
            "type": "Element",
            "sequential": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )


@dataclass
class NoteheadText:
    """The notehead-text type represents text that is displayed inside a
    notehead, as is done in some educational music.

    It is not needed for the numbers used in tablature or jianpu
    notation. The presence of a TAB or jianpu clefs is sufficient to
    indicate that numbers are used. The display-text and accidental-text
    elements allow display of fully formatted text and accidentals.
    """
    class Meta:
        name = "notehead-text"

    display_text: List[FormattedText] = field(
        default_factory=list,
        metadata={
            "name": "display-text",
            "type": "Element",
            "sequential": True,
        }
    )
    accidental_text: List[AccidentalText] = field(
        default_factory=list,
        metadata={
            "name": "accidental-text",
            "type": "Element",
            "sequential": True,
        }
    )


@dataclass
class Numeral:
    """The numeral type represents the Roman numeral or Nashville number part
    of a harmony.

    It requires that the key be specified in the encoding, either with a
    key or numeral-key element.

    :ivar numeral_root:
    :ivar numeral_alter: The numeral-alter element represents an
        alteration to the numeral-root, similar to the alter element for
        a pitch. The print-object attribute can be used to hide an
        alteration in cases such as when the MusicXML encoding of a 6 or
        7 numeral-root in a minor key requires an alteration that is not
        displayed. The location attribute indicates whether the
        alteration should appear to the left or the right of the
        numeral-root. It is left by default.
    :ivar numeral_key:
    """
    class Meta:
        name = "numeral"

    numeral_root: Optional[NumeralRoot] = field(
        default=None,
        metadata={
            "name": "numeral-root",
            "type": "Element",
            "required": True,
        }
    )
    numeral_alter: Optional[HarmonyAlter] = field(
        default=None,
        metadata={
            "name": "numeral-alter",
            "type": "Element",
        }
    )
    numeral_key: Optional[NumeralKey] = field(
        default=None,
        metadata={
            "name": "numeral-key",
            "type": "Element",
        }
    )


@dataclass
class PageLayout:
    """Page layout can be defined both in score-wide defaults and in the print
    element.

    Page margins are specified either for both even and odd pages, or
    via separate odd and even page number values. The type is not needed
    when used as part of a print element. If omitted when used in the
    defaults element, "both" is the default. If no page-layout element
    is present in the defaults element, default page layout values are
    chosen by the application. When used in the print element, the page-
    layout element affects the appearance of the current page only. All
    other pages use the default values as determined by the defaults
    element. If any child elements are missing from the page-layout
    element in a print element, the values determined by the defaults
    element are used there as well.
    """
    class Meta:
        name = "page-layout"

    page_height: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "page-height",
            "type": "Element",
        }
    )
    page_width: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "page-width",
            "type": "Element",
        }
    )
    page_margins: List[PageMargins] = field(
        default_factory=list,
        metadata={
            "name": "page-margins",
            "type": "Element",
            "max_occurs": 2,
        }
    )


@dataclass
class PartTranspose:
    """The child elements of the part-transpose type have the same meaning as
    for the transpose type.

    However that meaning applies to a transposed part created from the
    existing score file.

    :ivar diatonic: The diatonic element specifies the number of pitch
        steps needed to go from written to sounding pitch. This allows
        for correct spelling of enharmonic transpositions. This value
        does not include octave-change values; the values for both
        elements need to be added to the written pitch to get the
        correct sounding pitch.
    :ivar chromatic: The chromatic element represents the number of
        semitones needed to get from written to sounding pitch. This
        value does not include octave-change values; the values for both
        elements need to be added to the written pitch to get the
        correct sounding pitch.
    :ivar octave_change: The octave-change element indicates how many
        octaves to add to get from written pitch to sounding pitch. The
        octave-change element should be included when using
        transposition intervals of an octave or more, and should not be
        present for intervals of less than an octave.
    :ivar double: If the double element is present, it indicates that
        the music is doubled one octave from what is currently written.
    """
    class Meta:
        name = "part-transpose"

    diatonic: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    chromatic: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    octave_change: Optional[int] = field(
        default=None,
        metadata={
            "name": "octave-change",
            "type": "Element",
        }
    )
    double: Optional[Double] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Percussion:
    """The percussion element is used to define percussion pictogram symbols.

    Definitions for these symbols can be found in Kurt Stone's "Music
    Notation in the Twentieth Century" on pages 206-212 and 223. Some
    values are added to these based on how usage has evolved in the 30
    years since Stone's book was published.

    :ivar glass:
    :ivar metal:
    :ivar wood:
    :ivar pitched:
    :ivar membrane:
    :ivar effect:
    :ivar timpani:
    :ivar beater:
    :ivar stick:
    :ivar stick_location:
    :ivar other_percussion: The other-percussion element represents
        percussion pictograms not defined elsewhere.
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar enclosure:
    :ivar id:
    """
    class Meta:
        name = "percussion"

    glass: Optional[Glass] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    metal: Optional[Metal] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    wood: Optional[Wood] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    pitched: Optional[Pitched] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    membrane: Optional[Membrane] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    effect: Optional[Effect] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    timpani: Optional[Timpani] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    beater: Optional[Beater] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    stick: Optional[Stick] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    stick_location: Optional[StickLocation] = field(
        default=None,
        metadata={
            "name": "stick-location",
            "type": "Element",
        }
    )
    other_percussion: Optional[OtherText] = field(
        default=None,
        metadata={
            "name": "other-percussion",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    enclosure: Optional[EnclosureShape] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Root:
    """The root type indicates a pitch like C, D, E vs.

    a scale degree like 1, 2, 3. It is used with chord symbols in
    popular music. The root element has a root-step and optional root-
    alter element similar to the step and alter elements, but renamed to
    distinguish the different musical meanings.

    :ivar root_step:
    :ivar root_alter: The root-alter element represents the chromatic
        alteration of the root of the current chord within the harmony
        element. In some chord styles, the text for the root-step
        element may include root-alter information. In that case, the
        print-object attribute of the root-alter element can be set to
        no. The location attribute indicates whether the alteration
        should appear to the left or the right of the root-step; it is
        right by default.
    """
    class Meta:
        name = "root"

    root_step: Optional[RootStep] = field(
        default=None,
        metadata={
            "name": "root-step",
            "type": "Element",
            "required": True,
        }
    )
    root_alter: Optional[HarmonyAlter] = field(
        default=None,
        metadata={
            "name": "root-alter",
            "type": "Element",
        }
    )


@dataclass
class Scordatura:
    """Scordatura string tunings are represented by a series of accord
    elements, similar to the staff-tuning elements.

    Strings are numbered from high to low.
    """
    class Meta:
        name = "scordatura"

    accord: List[Accord] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Sound:
    """The sound element contains general playback parameters.

    They can stand alone within a part/measure, or be a component
    element within a direction. Tempo is expressed in quarter notes per
    minute. If 0, the sound-generating program should prompt the user at
    the time of compiling a sound (MIDI) file. Dynamics (or MIDI
    velocity) are expressed as a percentage of the default forte value
    (90 for MIDI 1.0). Dacapo indicates to go back to the beginning of
    the movement. When used it always has the value "yes". Segno and
    dalsegno are used for backwards jumps to a segno sign; coda and
    tocoda are used for forward jumps to a coda sign. If there are
    multiple jumps, the value of these parameters can be used to name
    and distinguish them. If segno or coda is used, the divisions
    attribute can also be used to indicate the number of divisions per
    quarter note. Otherwise sound and MIDI generating programs may have
    to recompute this. By default, a dalsegno or dacapo attribute
    indicates that the jump should occur the first time through, while a
    tocoda attribute indicates the jump should occur the second time
    through. The time that jumps occur can be changed by using the time-
    only attribute. The forward-repeat attribute indicates that a
    forward repeat sign is implied but not displayed. It is used for
    example in two-part forms with repeats, such as a minuet and trio
    where no repeat is displayed at the start of the trio. This usually
    occurs after a barline. When used it always has the value of "yes".
    The fine attribute follows the final note or rest in a movement with
    a da capo or dal segno direction. If numeric, the value represents
    the actual duration of the final note or rest, which can be
    ambiguous in written notation and different among parts and voices.
    The value may also be "yes" to indicate no change to the final
    duration. If the sound element applies only particular times through
    a repeat, the time-only attribute indicates which times to apply the
    sound element. Pizzicato in a sound element effects all following
    notes. Yes indicates pizzicato, no indicates arco. The pan and
    elevation attributes are deprecated in Version 2.0. The pan and
    elevation elements in the midi-instrument element should be used
    instead. The meaning of the pan and elevation attributes is the same
    as for the pan and elevation elements. If both are present, the mid-
    instrument elements take priority. The damper-pedal, soft-pedal, and
    sostenuto-pedal attributes effect playback of the three common piano
    pedals and their MIDI controller equivalents. The yes value
    indicates the pedal is depressed; no indicates the pedal is
    released. A numeric value from 0 to 100 may also be used for half
    pedaling. This value is the percentage that the pedal is depressed.
    A value of 0 is equivalent to no, and a value of 100 is equivalent
    to yes. Instrument changes, MIDI devices, MIDI instruments, and
    playback techniques are changed using the instrument-change, midi-
    device, midi-instrument, and play elements. When there are multiple
    instances of these elements, they should be grouped together by
    instrument using the id attribute values. The offset element is used
    to indicate that the sound takes place offset from the current score
    position. If the sound element is a child of a direction element,
    the sound offset element overrides the direction offset element if
    both elements are present. Note that the offset reflects the
    intended musical position for the change in sound. It should not be
    used to compensate for latency issues in particular hardware
    configurations.
    """
    class Meta:
        name = "sound"

    instrument_change: List[InstrumentChange] = field(
        default_factory=list,
        metadata={
            "name": "instrument-change",
            "type": "Element",
            "sequential": True,
        }
    )
    midi_device: List[MidiDevice] = field(
        default_factory=list,
        metadata={
            "name": "midi-device",
            "type": "Element",
            "sequential": True,
        }
    )
    midi_instrument: List[MidiInstrument] = field(
        default_factory=list,
        metadata={
            "name": "midi-instrument",
            "type": "Element",
            "sequential": True,
        }
    )
    play: List[Play] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    swing: Optional[Swing] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    offset: Optional[Offset] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    tempo: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
        }
    )
    dynamics: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
        }
    )
    dacapo: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    segno: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    dalsegno: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    coda: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    tocoda: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    divisions: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    forward_repeat: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "forward-repeat",
            "type": "Attribute",
        }
    )
    fine: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )
    pizzicato: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    pan: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    elevation: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("-180"),
            "max_inclusive": Decimal("180"),
        }
    )
    damper_pedal: Optional[Union[YesNo, Decimal]] = field(
        default=None,
        metadata={
            "name": "damper-pedal",
            "type": "Attribute",
        }
    )
    soft_pedal: Optional[Union[YesNo, Decimal]] = field(
        default=None,
        metadata={
            "name": "soft-pedal",
            "type": "Attribute",
        }
    )
    sostenuto_pedal: Optional[Union[YesNo, Decimal]] = field(
        default=None,
        metadata={
            "name": "sostenuto-pedal",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class StaffDetails:
    """The staff-details element is used to indicate different types of staves.

    The optional number attribute specifies the staff number from top to
    bottom on the system, as with clef. The print-object attribute is
    used to indicate when a staff is not printed in a part, usually in
    large scores where empty parts are omitted. It is yes by default. If
    print-spacing is yes while print-object is no, the score is printed
    in cutaway format where vertical space is left for the empty part.

    :ivar staff_type:
    :ivar staff_lines: The staff-lines element specifies the number of
        lines and is usually used for a non 5-line staff. If the staff-
        lines element is present, the appearance of each line may be
        individually specified with a line-detail element.
    :ivar line_detail:
    :ivar staff_tuning:
    :ivar capo: The capo element indicates at which fret a capo should
        be placed on a fretted instrument. This changes the open tuning
        of the strings specified by staff-tuning by the specified number
        of half-steps.
    :ivar staff_size:
    :ivar number:
    :ivar show_frets:
    :ivar print_object:
    :ivar print_spacing:
    """
    class Meta:
        name = "staff-details"

    staff_type: Optional[StaffType] = field(
        default=None,
        metadata={
            "name": "staff-type",
            "type": "Element",
        }
    )
    staff_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "staff-lines",
            "type": "Element",
        }
    )
    line_detail: List[LineDetail] = field(
        default_factory=list,
        metadata={
            "name": "line-detail",
            "type": "Element",
        }
    )
    staff_tuning: List[StaffTuning] = field(
        default_factory=list,
        metadata={
            "name": "staff-tuning",
            "type": "Element",
        }
    )
    capo: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    staff_size: Optional[StaffSize] = field(
        default=None,
        metadata={
            "name": "staff-size",
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    show_frets: Optional[ShowFrets] = field(
        default=None,
        metadata={
            "name": "show-frets",
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    print_spacing: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-spacing",
            "type": "Attribute",
        }
    )


@dataclass
class StrongAccent(EmptyPlacement):
    """The strong-accent type indicates a vertical accent mark.

    The type attribute indicates if the point of the accent is down or
    up.
    """
    class Meta:
        name = "strong-accent"

    type: UpDown = field(
        default=UpDown.UP,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class SystemDividers:
    """The system-dividers element indicates the presence or absence of system
    dividers (also known as system separation marks) between systems displayed
    on the same page.

    Dividers on the left and right side of the page are controlled by
    the left-divider and right-divider elements respectively. The
    default vertical position is half the system-distance value from the
    top of the system that is below the divider. The default horizontal
    position is the left and right system margin, respectively. When
    used in the print element, the system-dividers element affects the
    dividers that would appear between the current system and the
    previous system.
    """
    class Meta:
        name = "system-dividers"

    left_divider: Optional[EmptyPrintObjectStyleAlign] = field(
        default=None,
        metadata={
            "name": "left-divider",
            "type": "Element",
            "required": True,
        }
    )
    right_divider: Optional[EmptyPrintObjectStyleAlign] = field(
        default=None,
        metadata={
            "name": "right-divider",
            "type": "Element",
            "required": True,
        }
    )


@dataclass
class Time:
    """Time signatures are represented by the beats element for the numerator
    and the beat-type element for the denominator.

    The symbol attribute is used to indicate common and cut time symbols
    as well as a single number display. Multiple pairs of beat and beat-
    type elements are used for composite time signatures with multiple
    denominators, such as 2/4 + 3/8. A composite such as 3+2/8 requires
    only one beat/beat-type pair. The print-object attribute allows a
    time signature to be specified but not printed, as is the case for
    excerpts from the middle of a score. The value is "yes" if not
    present. The optional number attribute refers to staff numbers
    within the part. If absent, the time signature applies to all staves
    in the part.

    :ivar beats: The beats element indicates the number of beats, as
        found in the numerator of a time signature.
    :ivar beat_type: The beat-type element indicates the beat unit, as
        found in the denominator of a time signature.
    :ivar interchangeable:
    :ivar senza_misura: A senza-misura element explicitly indicates that
        no time signature is present. The optional element content
        indicates the symbol to be used, if any, such as an X. The time
        element's symbol attribute is not used when a senza-misura
        element is present.
    :ivar number:
    :ivar symbol:
    :ivar separator:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar print_object:
    :ivar id:
    """
    class Meta:
        name = "time"

    beats: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    beat_type: List[str] = field(
        default_factory=list,
        metadata={
            "name": "beat-type",
            "type": "Element",
            "sequential": True,
        }
    )
    interchangeable: Optional[Interchangeable] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    senza_misura: Optional[str] = field(
        default=None,
        metadata={
            "name": "senza-misura",
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    symbol: Optional[TimeSymbol] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    separator: Optional[TimeSeparator] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Transpose:
    """The transpose type represents what must be added to a written pitch to
    get a correct sounding pitch.

    The optional number attribute refers to staff numbers, from top to
    bottom on the system. If absent, the transposition applies to all
    staves in the part. Per-staff transposition is most often used in
    parts that represent multiple instruments.

    :ivar diatonic: The diatonic element specifies the number of pitch
        steps needed to go from written to sounding pitch. This allows
        for correct spelling of enharmonic transpositions. This value
        does not include octave-change values; the values for both
        elements need to be added to the written pitch to get the
        correct sounding pitch.
    :ivar chromatic: The chromatic element represents the number of
        semitones needed to get from written to sounding pitch. This
        value does not include octave-change values; the values for both
        elements need to be added to the written pitch to get the
        correct sounding pitch.
    :ivar octave_change: The octave-change element indicates how many
        octaves to add to get from written pitch to sounding pitch. The
        octave-change element should be included when using
        transposition intervals of an octave or more, and should not be
        present for intervals of less than an octave.
    :ivar double: If the double element is present, it indicates that
        the music is doubled one octave from what is currently written.
    :ivar number:
    :ivar id:
    """
    class Meta:
        name = "transpose"

    diatonic: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    chromatic: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    octave_change: Optional[int] = field(
        default=None,
        metadata={
            "name": "octave-change",
            "type": "Element",
        }
    )
    double: Optional[Double] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class TupletPortion:
    """The tuplet-portion type provides optional full control over tuplet
    specifications.

    It allows the number and note type (including dots) to be set for
    the actual and normal portions of a single tuplet. If any of these
    elements are absent, their values are based on the time-modification
    element.
    """
    class Meta:
        name = "tuplet-portion"

    tuplet_number: Optional[TupletNumber] = field(
        default=None,
        metadata={
            "name": "tuplet-number",
            "type": "Element",
        }
    )
    tuplet_type: Optional[TupletType] = field(
        default=None,
        metadata={
            "name": "tuplet-type",
            "type": "Element",
        }
    )
    tuplet_dot: List[TupletDot] = field(
        default_factory=list,
        metadata={
            "name": "tuplet-dot",
            "type": "Element",
        }
    )


@dataclass
class Work:
    """Works are optionally identified by number and title.

    The work type also may indicate a link to the opus document that
    composes multiple scores into a collection.

    :ivar work_number: The work-number element specifies the number of a
        work, such as its opus number.
    :ivar work_title: The work-title element specifies the title of a
        work, not including its opus or other work number.
    :ivar opus:
    """
    class Meta:
        name = "work"

    work_number: Optional[str] = field(
        default=None,
        metadata={
            "name": "work-number",
            "type": "Element",
        }
    )
    work_title: Optional[str] = field(
        default=None,
        metadata={
            "name": "work-title",
            "type": "Element",
        }
    )
    opus: Optional[Opus] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Articulations:
    """
    Articulations and accents are grouped together here.

    :ivar accent: The accent element indicates a regular horizontal
        accent mark.
    :ivar strong_accent: The strong-accent element indicates a vertical
        accent mark.
    :ivar staccato: The staccato element is used for a dot articulation,
        as opposed to a stroke or a wedge.
    :ivar tenuto: The tenuto element indicates a tenuto line symbol.
    :ivar detached_legato: The detached-legato element indicates the
        combination of a tenuto line and staccato dot symbol.
    :ivar staccatissimo: The staccatissimo element is used for a wedge
        articulation, as opposed to a dot or a stroke.
    :ivar spiccato: The spiccato element is used for a stroke
        articulation, as opposed to a dot or a wedge.
    :ivar scoop: The scoop element is an indeterminate slide attached to
        a single note. The scoop appears before the main note and comes
        from below the main pitch.
    :ivar plop: The plop element is an indeterminate slide attached to a
        single note. The plop appears before the main note and comes
        from above the main pitch.
    :ivar doit: The doit element is an indeterminate slide attached to a
        single note. The doit appears after the main note and goes above
        the main pitch.
    :ivar falloff: The falloff element is an indeterminate slide
        attached to a single note. The falloff appears after the main
        note and goes below the main pitch.
    :ivar breath_mark:
    :ivar caesura:
    :ivar stress: The stress element indicates a stressed note.
    :ivar unstress: The unstress element indicates an unstressed note.
        It is often notated using a u-shaped symbol.
    :ivar soft_accent: The soft-accent element indicates a soft accent
        that is not as heavy as a normal accent. It is often notated as
        &lt;&gt;. It can be combined with other articulations to
        implement the first eight symbols in the SMuFL Articulation
        supplement range.
    :ivar other_articulation: The other-articulation element is used to
        define any articulations not yet in the MusicXML format. The
        smufl attribute can be used to specify a particular
        articulation, allowing application interoperability without
        requiring every SMuFL articulation to have a MusicXML element
        equivalent. Using the other-articulation element without the
        smufl attribute allows for extended representation, though
        without application interoperability.
    :ivar id:
    """
    class Meta:
        name = "articulations"

    accent: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    strong_accent: List[StrongAccent] = field(
        default_factory=list,
        metadata={
            "name": "strong-accent",
            "type": "Element",
        }
    )
    staccato: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    tenuto: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    detached_legato: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "detached-legato",
            "type": "Element",
        }
    )
    staccatissimo: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    spiccato: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    scoop: List[EmptyLine] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    plop: List[EmptyLine] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    doit: List[EmptyLine] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    falloff: List[EmptyLine] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    breath_mark: List[BreathMark] = field(
        default_factory=list,
        metadata={
            "name": "breath-mark",
            "type": "Element",
        }
    )
    caesura: List[Caesura] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    stress: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    unstress: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    soft_accent: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "soft-accent",
            "type": "Element",
        }
    )
    other_articulation: List[OtherPlacementText] = field(
        default_factory=list,
        metadata={
            "name": "other-articulation",
            "type": "Element",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class FiguredBass:
    """The figured-bass element represents figured bass notation.

    Figured bass elements take their position from the first regular
    note (not a grace note or chord note) that follows in score order.
    The optional duration element is used to indicate changes of figures
    under a note. Figures are ordered from top to bottom. The value of
    parentheses is "no" if not present.

    :ivar figure:
    :ivar duration: Duration is a positive number specified in division
        units. This is the intended duration vs. notated duration (for
        instance, differences in dotted notes in Baroque-era music).
        Differences in duration specific to an interpretation or
        performance should be represented using the note element's
        attack and release attributes. The duration element moves the
        musical position when used in backup elements, forward elements,
        and note elements that do not contain a chord child element.
    :ivar footnote:
    :ivar level:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar placement:
    :ivar print_object:
    :ivar print_dot:
    :ivar print_spacing:
    :ivar print_lyric:
    :ivar parentheses:
    :ivar id:
    """
    class Meta:
        name = "figured-bass"

    figure: List[Figure] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )
    duration: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "min_exclusive": Decimal("0"),
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    print_dot: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-dot",
            "type": "Attribute",
        }
    )
    print_spacing: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-spacing",
            "type": "Attribute",
        }
    )
    print_lyric: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-lyric",
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class ForPart:
    """The for-part type is used in a concert score to indicate the
    transposition for a transposed part created from that score.

    It is only used in score files that contain a concert-score element
    in the defaults. This allows concert scores with transposed parts to
    be represented in a single uncompressed MusicXML file. The optional
    number attribute refers to staff numbers, from top to bottom on the
    system. If absent, the child elements apply to all staves in the
    created part.

    :ivar part_clef: The part-clef element is used for transpositions
        that also include a change of clef, as for instruments such as
        bass clarinet.
    :ivar part_transpose: The chromatic element in a part-transpose
        element will usually have a non-zero value, since octave
        transpositions can be represented in concert scores using the
        transpose element.
    :ivar number:
    :ivar id:
    """
    class Meta:
        name = "for-part"

    part_clef: Optional[PartClef] = field(
        default=None,
        metadata={
            "name": "part-clef",
            "type": "Element",
        }
    )
    part_transpose: Optional[PartTranspose] = field(
        default=None,
        metadata={
            "name": "part-transpose",
            "type": "Element",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Frame:
    """The frame type represents a frame or fretboard diagram used together
    with a chord symbol.

    The representation is based on the NIFF guitar grid with additional
    information. The frame type's unplayed attribute indicates what to
    display above a string that has no associated frame-note element.
    Typical values are x and the empty string. If the attribute is not
    present, the display of the unplayed string is application-defined.

    :ivar frame_strings: The frame-strings element gives the overall
        size of the frame in vertical lines (strings).
    :ivar frame_frets: The frame-frets element gives the overall size of
        the frame in horizontal spaces (frets).
    :ivar first_fret:
    :ivar frame_note:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar height:
    :ivar width:
    :ivar unplayed:
    :ivar id:
    """
    class Meta:
        name = "frame"

    frame_strings: Optional[int] = field(
        default=None,
        metadata={
            "name": "frame-strings",
            "type": "Element",
            "required": True,
        }
    )
    frame_frets: Optional[int] = field(
        default=None,
        metadata={
            "name": "frame-frets",
            "type": "Element",
            "required": True,
        }
    )
    first_fret: Optional[FirstFret] = field(
        default=None,
        metadata={
            "name": "first-fret",
            "type": "Element",
        }
    )
    frame_note: List[FrameNote] = field(
        default_factory=list,
        metadata={
            "name": "frame-note",
            "type": "Element",
            "min_occurs": 1,
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[ValignImage] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    height: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    width: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    unplayed: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Identification:
    """Identification contains basic metadata about the score.

    It includes information that may apply at a score-wide, movement-
    wide, or part-wide level. The creator, rights, source, and relation
    elements are based on Dublin Core.

    :ivar creator: The creator element is borrowed from Dublin Core. It
        is used for the creators of the score. The type attribute is
        used to distinguish different creative contributions. Thus,
        there can be multiple creators within an identification.
        Standard type values are composer, lyricist, and arranger. Other
        type values may be used for different types of creative roles.
        The type attribute should usually be used even if there is just
        a single creator element. The MusicXML format does not use the
        creator / contributor distinction from Dublin Core.
    :ivar rights: The rights element is borrowed from Dublin Core. It
        contains copyright and other intellectual property notices.
        Words, music, and derivatives can have different types, so
        multiple rights elements with different type attributes are
        supported. Standard type values are music, words, and
        arrangement, but other types may be used. The type attribute is
        only needed when there are multiple rights elements.
    :ivar encoding:
    :ivar source: The source for the music that is encoded. This is
        similar to the Dublin Core source element.
    :ivar relation: A related resource for the music that is encoded.
        This is similar to the Dublin Core relation element. Standard
        type values are music, words, and arrangement, but other types
        may be used.
    :ivar miscellaneous:
    """
    class Meta:
        name = "identification"

    creator: List[TypedText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    rights: List[TypedText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    encoding: Optional[Encoding] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    source: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    relation: List[TypedText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    miscellaneous: Optional[Miscellaneous] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class MetronomeNote:
    """
    The metronome-note type defines the appearance of a note within a metric
    relationship mark.

    :ivar metronome_type: The metronome-type element works like the type
        element in defining metric relationships.
    :ivar metronome_dot: The metronome-dot element works like the dot
        element in defining metric relationships.
    :ivar metronome_beam:
    :ivar metronome_tied:
    :ivar metronome_tuplet:
    """
    class Meta:
        name = "metronome-note"

    metronome_type: Optional[NoteTypeValue] = field(
        default=None,
        metadata={
            "name": "metronome-type",
            "type": "Element",
            "required": True,
        }
    )
    metronome_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "metronome-dot",
            "type": "Element",
        }
    )
    metronome_beam: List[MetronomeBeam] = field(
        default_factory=list,
        metadata={
            "name": "metronome-beam",
            "type": "Element",
        }
    )
    metronome_tied: Optional[MetronomeTied] = field(
        default=None,
        metadata={
            "name": "metronome-tied",
            "type": "Element",
        }
    )
    metronome_tuplet: Optional[MetronomeTuplet] = field(
        default=None,
        metadata={
            "name": "metronome-tuplet",
            "type": "Element",
        }
    )


@dataclass
class Ornaments:
    """Ornaments can be any of several types, followed optionally by
    accidentals.

    The accidental-mark element's content is represented the same as an
    accidental element, but with a different name to reflect the
    different musical meaning.

    :ivar trill_mark: The trill-mark element represents the trill-mark
        symbol.
    :ivar turn: The turn element is the normal turn shape which goes up
        then down.
    :ivar delayed_turn: The delayed-turn element indicates a normal turn
        that is delayed until the end of the current note.
    :ivar inverted_turn: The inverted-turn element has the shape which
        goes down and then up.
    :ivar delayed_inverted_turn: The delayed-inverted-turn element
        indicates an inverted turn that is delayed until the end of the
        current note.
    :ivar vertical_turn: The vertical-turn element has the turn symbol
        shape arranged vertically going from upper left to lower right.
    :ivar inverted_vertical_turn: The inverted-vertical-turn element has
        the turn symbol shape arranged vertically going from upper right
        to lower left.
    :ivar shake: The shake element has a similar appearance to an
        inverted-mordent element.
    :ivar wavy_line:
    :ivar mordent: The mordent element represents the sign with the
        vertical line. The choice of which mordent sign is inverted
        differs between MusicXML and SMuFL. The long attribute is "no"
        by default.
    :ivar inverted_mordent: The inverted-mordent element represents the
        sign without the vertical line. The choice of which mordent is
        inverted differs between MusicXML and SMuFL. The long attribute
        is "no" by default.
    :ivar schleifer: The name for this ornament is based on the German,
        to avoid confusion with the more common slide element defined
        earlier.
    :ivar tremolo:
    :ivar haydn: The haydn element represents the Haydn ornament. This
        is defined in SMuFL as ornamentHaydn.
    :ivar other_ornament: The other-ornament element is used to define
        any ornaments not yet in the MusicXML format. The smufl
        attribute can be used to specify a particular ornament, allowing
        application interoperability without requiring every SMuFL
        ornament to have a MusicXML element equivalent. Using the other-
        ornament element without the smufl attribute allows for extended
        representation, though without application interoperability.
    :ivar accidental_mark:
    :ivar id:
    """
    class Meta:
        name = "ornaments"

    trill_mark: List[EmptyTrillSound] = field(
        default_factory=list,
        metadata={
            "name": "trill-mark",
            "type": "Element",
            "sequential": True,
        }
    )
    turn: List[HorizontalTurn] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    delayed_turn: List[HorizontalTurn] = field(
        default_factory=list,
        metadata={
            "name": "delayed-turn",
            "type": "Element",
            "sequential": True,
        }
    )
    inverted_turn: List[HorizontalTurn] = field(
        default_factory=list,
        metadata={
            "name": "inverted-turn",
            "type": "Element",
            "sequential": True,
        }
    )
    delayed_inverted_turn: List[HorizontalTurn] = field(
        default_factory=list,
        metadata={
            "name": "delayed-inverted-turn",
            "type": "Element",
            "sequential": True,
        }
    )
    vertical_turn: List[EmptyTrillSound] = field(
        default_factory=list,
        metadata={
            "name": "vertical-turn",
            "type": "Element",
            "sequential": True,
        }
    )
    inverted_vertical_turn: List[EmptyTrillSound] = field(
        default_factory=list,
        metadata={
            "name": "inverted-vertical-turn",
            "type": "Element",
            "sequential": True,
        }
    )
    shake: List[EmptyTrillSound] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    wavy_line: List[WavyLine] = field(
        default_factory=list,
        metadata={
            "name": "wavy-line",
            "type": "Element",
            "sequential": True,
        }
    )
    mordent: List[Mordent] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    inverted_mordent: List[Mordent] = field(
        default_factory=list,
        metadata={
            "name": "inverted-mordent",
            "type": "Element",
            "sequential": True,
        }
    )
    schleifer: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    tremolo: List[Tremolo] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    haydn: List[EmptyTrillSound] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    other_ornament: List[OtherPlacementText] = field(
        default_factory=list,
        metadata={
            "name": "other-ornament",
            "type": "Element",
            "sequential": True,
        }
    )
    accidental_mark: List[AccidentalMark] = field(
        default_factory=list,
        metadata={
            "name": "accidental-mark",
            "type": "Element",
            "sequential": True,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PartGroup:
    """The part-group element indicates groupings of parts in the score,
    usually indicated by braces and brackets.

    Braces that are used for multi-staff parts should be defined in the
    attributes element for that part. The part-group start element
    appears before the first score-part in the group. The part-group
    stop element appears after the last score-part in the group. The
    number attribute is used to distinguish overlapping and nested part-
    groups, not the sequence of groups. As with parts, groups can have a
    name and abbreviation. Values for the child elements are ignored at
    the stop of a group. A part-group element is not needed for a single
    multi-staff part. By default, multi-staff parts include a brace
    symbol and (if appropriate given the bar-style) common barlines. The
    symbol formatting for a multi-staff part can be more fully specified
    using the part-symbol element.

    :ivar group_name:
    :ivar group_name_display: Formatting specified in the group-name-
        display element overrides formatting specified in the group-name
        element.
    :ivar group_abbreviation:
    :ivar group_abbreviation_display: Formatting specified in the group-
        abbreviation-display element overrides formatting specified in
        the group-abbreviation element.
    :ivar group_symbol:
    :ivar group_barline:
    :ivar group_time: The group-time element indicates that the
        displayed time signatures should stretch across all parts and
        staves in the group.
    :ivar footnote:
    :ivar level:
    :ivar type:
    :ivar number:
    """
    class Meta:
        name = "part-group"

    group_name: Optional[GroupName] = field(
        default=None,
        metadata={
            "name": "group-name",
            "type": "Element",
        }
    )
    group_name_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "group-name-display",
            "type": "Element",
        }
    )
    group_abbreviation: Optional[GroupName] = field(
        default=None,
        metadata={
            "name": "group-abbreviation",
            "type": "Element",
        }
    )
    group_abbreviation_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "group-abbreviation-display",
            "type": "Element",
        }
    )
    group_symbol: Optional[GroupSymbol] = field(
        default=None,
        metadata={
            "name": "group-symbol",
            "type": "Element",
        }
    )
    group_barline: Optional[GroupBarline] = field(
        default=None,
        metadata={
            "name": "group-barline",
            "type": "Element",
        }
    )
    group_time: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "group-time",
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: str = field(
        default="1",
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class SystemLayout:
    """A system is a group of staves that are read and played simultaneously.

    System layout includes left and right margins and the vertical
    distance from the previous system. The system distance is measured
    from the bottom line of the previous system to the top line of the
    current system. It is ignored for the first system on a page. The
    top system distance is measured from the page's top margin to the
    top line of the first system. It is ignored for all but the first
    system on a page. Sometimes the sum of measure widths in a system
    may not equal the system width specified by the layout elements due
    to roundoff or other errors. The behavior when reading MusicXML
    files in these cases is application-dependent. For instance,
    applications may find that the system layout data is more reliable
    than the sum of the measure widths, and adjust the measure widths
    accordingly. When used in the defaults element, the system-layout
    element defines a default appearance for all systems in the score.
    If no system-layout element is present in the defaults element,
    default system layout values are chosen by the application. When
    used in the print element, the system-layout element affects the
    appearance of the current system only. All other systems use the
    default values as determined by the defaults element. If any child
    elements are missing from the system-layout element in a print
    element, the values determined by the defaults element are used
    there as well. This type of system-layout element need only be read
    from or written to the first visible part in the score.
    """
    class Meta:
        name = "system-layout"

    system_margins: Optional[SystemMargins] = field(
        default=None,
        metadata={
            "name": "system-margins",
            "type": "Element",
        }
    )
    system_distance: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "system-distance",
            "type": "Element",
        }
    )
    top_system_distance: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "top-system-distance",
            "type": "Element",
        }
    )
    system_dividers: Optional[SystemDividers] = field(
        default=None,
        metadata={
            "name": "system-dividers",
            "type": "Element",
        }
    )


@dataclass
class Technical:
    """
    Technical indications give performance information for individual
    instruments.

    :ivar up_bow: The up-bow element represents the symbol that is used
        both for up-bowing on bowed instruments, and up-stroke on
        plucked instruments.
    :ivar down_bow: The down-bow element represents the symbol that is
        used both for down-bowing on bowed instruments, and down-stroke
        on plucked instruments.
    :ivar harmonic:
    :ivar open_string: The open-string element represents the zero-
        shaped open string symbol.
    :ivar thumb_position: The thumb-position element represents the
        thumb position symbol. This is a circle with a line, where the
        line does not come within the circle. It is distinct from the
        snap pizzicato symbol, where the line comes inside the circle.
    :ivar fingering:
    :ivar pluck: The pluck element is used to specify the plucking
        fingering on a fretted instrument, where the fingering element
        refers to the fretting fingering. Typical values are p, i, m, a
        for pulgar/thumb, indicio/index, medio/middle, and anular/ring
        fingers.
    :ivar double_tongue: The double-tongue element represents the double
        tongue symbol (two dots arranged horizontally).
    :ivar triple_tongue: The triple-tongue element represents the triple
        tongue symbol (three dots arranged horizontally).
    :ivar stopped: The stopped element represents the stopped symbol,
        which looks like a plus sign. The smufl attribute distinguishes
        different SMuFL glyphs that have a similar appearance such as
        handbellsMalletBellSuspended and guitarClosePedal. If not
        present, the default glyph is brassMuteClosed.
    :ivar snap_pizzicato: The snap-pizzicato element represents the snap
        pizzicato symbol. This is a circle with a line, where the line
        comes inside the circle. It is distinct from the thumb-position
        symbol, where the line does not come inside the circle.
    :ivar fret:
    :ivar string:
    :ivar hammer_on:
    :ivar pull_off:
    :ivar bend:
    :ivar tap:
    :ivar heel:
    :ivar toe:
    :ivar fingernails: The fingernails element is used in notation for
        harp and other plucked string instruments.
    :ivar hole:
    :ivar arrow:
    :ivar handbell:
    :ivar brass_bend: The brass-bend element represents the u-shaped
        bend symbol used in brass notation, distinct from the bend
        element used in guitar music.
    :ivar flip: The flip element represents the flip symbol used in
        brass notation.
    :ivar smear: The smear element represents the tilde-shaped smear
        symbol used in brass notation.
    :ivar open: The open element represents the open symbol, which looks
        like a circle. The smufl attribute can be used to distinguish
        different SMuFL glyphs that have a similar appearance such as
        brassMuteOpen and guitarOpenPedal. If not present, the default
        glyph is brassMuteOpen.
    :ivar half_muted: The half-muted element represents the half-muted
        symbol, which looks like a circle with a plus sign inside. The
        smufl attribute can be used to distinguish different SMuFL
        glyphs that have a similar appearance such as
        brassMuteHalfClosed and guitarHalfOpenPedal. If not present, the
        default glyph is brassMuteHalfClosed.
    :ivar harmon_mute:
    :ivar golpe: The golpe element represents the golpe symbol that is
        used for tapping the pick guard in guitar music.
    :ivar other_technical: The other-technical element is used to define
        any technical indications not yet in the MusicXML format. The
        smufl attribute can be used to specify a particular glyph,
        allowing application interoperability without requiring every
        SMuFL technical indication to have a MusicXML element
        equivalent. Using the other-technical element without the smufl
        attribute allows for extended representation, though without
        application interoperability.
    :ivar id:
    """
    class Meta:
        name = "technical"

    up_bow: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "up-bow",
            "type": "Element",
        }
    )
    down_bow: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "down-bow",
            "type": "Element",
        }
    )
    harmonic: List[Harmonic] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    open_string: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "open-string",
            "type": "Element",
        }
    )
    thumb_position: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "thumb-position",
            "type": "Element",
        }
    )
    fingering: List[Fingering] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    pluck: List[PlacementText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    double_tongue: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "double-tongue",
            "type": "Element",
        }
    )
    triple_tongue: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "triple-tongue",
            "type": "Element",
        }
    )
    stopped: List[EmptyPlacementSmufl] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    snap_pizzicato: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "snap-pizzicato",
            "type": "Element",
        }
    )
    fret: List[Fret] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    string: List[String] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    hammer_on: List[HammerOnPullOff] = field(
        default_factory=list,
        metadata={
            "name": "hammer-on",
            "type": "Element",
        }
    )
    pull_off: List[HammerOnPullOff] = field(
        default_factory=list,
        metadata={
            "name": "pull-off",
            "type": "Element",
        }
    )
    bend: List[Bend] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    tap: List[Tap] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    heel: List[HeelToe] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    toe: List[HeelToe] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    fingernails: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    hole: List[Hole] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    arrow: List[Arrow] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    handbell: List[Handbell] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    brass_bend: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "name": "brass-bend",
            "type": "Element",
        }
    )
    flip: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    smear: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    open: List[EmptyPlacementSmufl] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    half_muted: List[EmptyPlacementSmufl] = field(
        default_factory=list,
        metadata={
            "name": "half-muted",
            "type": "Element",
        }
    )
    harmon_mute: List[HarmonMute] = field(
        default_factory=list,
        metadata={
            "name": "harmon-mute",
            "type": "Element",
        }
    )
    golpe: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    other_technical: List[OtherPlacementText] = field(
        default_factory=list,
        metadata={
            "name": "other-technical",
            "type": "Element",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Tuplet:
    """A tuplet element is present when a tuplet is to be displayed
    graphically, in addition to the sound data provided by the time-
    modification elements.

    The number attribute is used to distinguish nested tuplets. The
    bracket attribute is used to indicate the presence of a bracket. If
    unspecified, the results are implementation-dependent. The line-
    shape attribute is used to specify whether the bracket is straight
    or in the older curved or slurred style. It is straight by default.
    Whereas a time-modification element shows how the cumulative,
    sounding effect of tuplets and double-note tremolos compare to the
    written note type, the tuplet element describes how this is
    displayed. The tuplet element also provides more detailed
    representation information than the time-modification element, and
    is needed to represent nested tuplets and other complex tuplets
    accurately. The show-number attribute is used to display either the
    number of actual notes, the number of both actual and normal notes,
    or neither. It is actual by default. The show-type attribute is used
    to display either the actual type, both the actual and normal types,
    or neither. It is none by default.

    :ivar tuplet_actual: The tuplet-actual element provide optional full
        control over how the actual part of the tuplet is displayed,
        including number and note type (with dots). If any of these
        elements are absent, their values are based on the time-
        modification element.
    :ivar tuplet_normal: The tuplet-normal element provide optional full
        control over how the normal part of the tuplet is displayed,
        including number and note type (with dots). If any of these
        elements are absent, their values are based on the time-
        modification element.
    :ivar type:
    :ivar number:
    :ivar bracket:
    :ivar show_number:
    :ivar show_type:
    :ivar line_shape:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar placement:
    :ivar id:
    """
    class Meta:
        name = "tuplet"

    tuplet_actual: Optional[TupletPortion] = field(
        default=None,
        metadata={
            "name": "tuplet-actual",
            "type": "Element",
        }
    )
    tuplet_normal: Optional[TupletPortion] = field(
        default=None,
        metadata={
            "name": "tuplet-normal",
            "type": "Element",
        }
    )
    type: Optional[StartStop] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": 1,
            "max_inclusive": 16,
        }
    )
    bracket: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    show_number: Optional[ShowTuplet] = field(
        default=None,
        metadata={
            "name": "show-number",
            "type": "Attribute",
        }
    )
    show_type: Optional[ShowTuplet] = field(
        default=None,
        metadata={
            "name": "show-type",
            "type": "Attribute",
        }
    )
    line_shape: Optional[LineShape] = field(
        default=None,
        metadata={
            "name": "line-shape",
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Attributes:
    """The attributes element contains musical information that typically
    changes on measure boundaries.

    This includes key and time signatures, clefs, transpositions, and
    staving. When attributes are changed mid-measure, it affects the
    music in score order, not in MusicXML document order.

    :ivar footnote:
    :ivar level:
    :ivar divisions: Musical notation duration is commonly represented
        as fractions. The divisions element indicates how many divisions
        per quarter note are used to indicate a note's duration. For
        example, if duration = 1 and divisions = 2, this is an eighth
        note duration. Duration and divisions are used directly for
        generating sound output, so they must be chosen to take tuplets
        into account. Using a divisions element lets us use just one
        number to represent a duration for each note in the score, while
        retaining the full power of a fractional representation. If
        maximum compatibility with Standard MIDI 1.0 files is important,
        do not have the divisions value exceed 16383.
    :ivar key: The key element represents a key signature. Both
        traditional and non-traditional key signatures are supported.
        The optional number attribute refers to staff numbers. If
        absent, the key signature applies to all staves in the part.
    :ivar time: Time signatures are represented by the beats element for
        the numerator and the beat-type element for the denominator.
    :ivar staves: The staves element is used if there is more than one
        staff represented in the given part (e.g., 2 staves for typical
        piano parts). If absent, a value of 1 is assumed. Staves are
        ordered from top to bottom in a part in numerical order, with
        staff 1 above staff 2.
    :ivar part_symbol: The part-symbol element indicates how a symbol
        for a multi-staff part is indicated in the score.
    :ivar instruments: The instruments element is only used if more than
        one instrument is represented in the part (e.g., oboe I and II
        where they play together most of the time). If absent, a value
        of 1 is assumed.
    :ivar clef: Clefs are represented by a combination of sign, line,
        and clef-octave-change elements.
    :ivar staff_details: The staff-details element is used to indicate
        different types of staves.
    :ivar transpose: If the part is being encoded for a transposing
        instrument in written vs. concert pitch, the transposition must
        be encoded in the transpose element using the transpose type.
    :ivar for_part: The for-part element is used in a concert score to
        indicate the transposition for a transposed part created from
        that score. It is only used in score files that contain a
        concert-score element in the defaults. This allows concert
        scores with transposed parts to be represented in a single
        uncompressed MusicXML file.
    :ivar directive: Directives are like directions, but can be grouped
        together with attributes for convenience. This is typically used
        for tempo markings at the beginning of a piece of music. This
        element was deprecated in Version 2.0 in favor of the direction
        element's directive attribute. Language names come from ISO 639,
        with optional country subcodes from ISO 3166.
    :ivar measure_style: A measure-style indicates a special way to
        print partial to multiple measures within a part. This includes
        multiple rests over several measures, repeats of beats, single,
        or multiple measures, and use of slash notation.
    """
    class Meta:
        name = "attributes"

    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    divisions: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Element",
            "min_exclusive": Decimal("0"),
        }
    )
    key: List[Key] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    time: List[Time] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    staves: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    part_symbol: Optional[PartSymbol] = field(
        default=None,
        metadata={
            "name": "part-symbol",
            "type": "Element",
        }
    )
    instruments: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    clef: List[Clef] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    staff_details: List[StaffDetails] = field(
        default_factory=list,
        metadata={
            "name": "staff-details",
            "type": "Element",
        }
    )
    transpose: List[Transpose] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    for_part: List[ForPart] = field(
        default_factory=list,
        metadata={
            "name": "for-part",
            "type": "Element",
        }
    )
    directive: List["Attributes.Directive"] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    measure_style: List[MeasureStyle] = field(
        default_factory=list,
        metadata={
            "name": "measure-style",
            "type": "Element",
        }
    )

    @dataclass
    class Directive:
        value: str = field(
            default="",
            metadata={
                "required": True,
            }
        )
        default_x: Optional[Decimal] = field(
            default=None,
            metadata={
                "name": "default-x",
                "type": "Attribute",
            }
        )
        default_y: Optional[Decimal] = field(
            default=None,
            metadata={
                "name": "default-y",
                "type": "Attribute",
            }
        )
        relative_x: Optional[Decimal] = field(
            default=None,
            metadata={
                "name": "relative-x",
                "type": "Attribute",
            }
        )
        relative_y: Optional[Decimal] = field(
            default=None,
            metadata={
                "name": "relative-y",
                "type": "Attribute",
            }
        )
        font_family: Optional[str] = field(
            default=None,
            metadata={
                "name": "font-family",
                "type": "Attribute",
                "pattern": r"[^,]+(, ?[^,]+)*",
            }
        )
        font_style: Optional[FontStyle] = field(
            default=None,
            metadata={
                "name": "font-style",
                "type": "Attribute",
            }
        )
        font_size: Optional[Union[Decimal, CssFontSize]] = field(
            default=None,
            metadata={
                "name": "font-size",
                "type": "Attribute",
            }
        )
        font_weight: Optional[FontWeight] = field(
            default=None,
            metadata={
                "name": "font-weight",
                "type": "Attribute",
            }
        )
        color: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
            }
        )
        lang: Optional[Union[str, LangValue]] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "namespace": "http://www.w3.org/XML/1998/namespace",
            }
        )


@dataclass
class Defaults:
    """The defaults type specifies score-wide defaults for scaling; whether or
    not the file is a concert score; layout; and default values for the music
    font, word font, lyric font, and lyric language.

    Except for the concert-score element, if any defaults are missing,
    the choice of what to use is determined by the application.

    :ivar scaling:
    :ivar concert_score: The presence of a concert-score element
        indicates that a score is displayed in concert pitch. It is used
        for scores that contain parts for transposing instruments. A
        document with a concert-score element may not contain any
        transpose elements that have non-zero values for either the
        diatonic or chromatic elements. Concert scores may include
        octave transpositions, so transpose elements with a double
        element or a non-zero octave-change element value are permitted.
    :ivar page_layout:
    :ivar system_layout:
    :ivar staff_layout:
    :ivar appearance:
    :ivar music_font:
    :ivar word_font:
    :ivar lyric_font:
    :ivar lyric_language:
    """
    class Meta:
        name = "defaults"

    scaling: Optional[Scaling] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    concert_score: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "concert-score",
            "type": "Element",
        }
    )
    page_layout: Optional[PageLayout] = field(
        default=None,
        metadata={
            "name": "page-layout",
            "type": "Element",
        }
    )
    system_layout: Optional[SystemLayout] = field(
        default=None,
        metadata={
            "name": "system-layout",
            "type": "Element",
        }
    )
    staff_layout: List[StaffLayout] = field(
        default_factory=list,
        metadata={
            "name": "staff-layout",
            "type": "Element",
        }
    )
    appearance: Optional[Appearance] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    music_font: Optional[EmptyFont] = field(
        default=None,
        metadata={
            "name": "music-font",
            "type": "Element",
        }
    )
    word_font: Optional[EmptyFont] = field(
        default=None,
        metadata={
            "name": "word-font",
            "type": "Element",
        }
    )
    lyric_font: List[LyricFont] = field(
        default_factory=list,
        metadata={
            "name": "lyric-font",
            "type": "Element",
        }
    )
    lyric_language: List[LyricLanguage] = field(
        default_factory=list,
        metadata={
            "name": "lyric-language",
            "type": "Element",
        }
    )


@dataclass
class Harmony:
    """The harmony type represents harmony analysis, including chord symbols in
    popular music as well as functional harmony analysis in classical music.

    If there are alternate harmonies possible, this can be specified
    using multiple harmony elements differentiated by type. Explicit
    harmonies have all note present in the music; implied have some
    notes missing but implied; alternate represents alternate analyses.
    The print-object attribute controls whether or not anything is
    printed due to the harmony element. The print-frame attribute
    controls printing of a frame or fretboard diagram. The print-style
    attribute group sets the default for the harmony, but individual
    elements can override this with their own print-style values. The
    arrangement attribute specifies how multiple harmony-chord groups
    are arranged relative to each other. Harmony-chords with vertical
    arrangement are separated by horizontal lines. Harmony-chords with
    diagonal or horizontal arrangement are separated by diagonal lines
    or slashes.

    :ivar root:
    :ivar numeral:
    :ivar function: The function element represents classical functional
        harmony with an indication like I, II, III rather than C, D, E.
        It represents the Roman numeral part of a functional harmony
        rather than the complete function itself. It has been deprecated
        as of MusicXML 4.0 in favor of the numeral element.
    :ivar kind:
    :ivar inversion:
    :ivar bass:
    :ivar degree:
    :ivar frame:
    :ivar offset:
    :ivar footnote:
    :ivar level:
    :ivar staff: Staff assignment is only needed for music notated on
        multiple staves. Used by both notes and directions. Staff values
        are numbers, with 1 referring to the top-most staff in a part.
    :ivar type:
    :ivar print_object:
    :ivar print_frame:
    :ivar arrangement:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar placement:
    :ivar system:
    :ivar id:
    """
    class Meta:
        name = "harmony"

    root: List[Root] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    numeral: List[Numeral] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    function: List[StyleText] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    kind: List[Kind] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequential": True,
        }
    )
    inversion: List[Inversion] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    bass: List[Bass] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    degree: List[Degree] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    frame: Optional[Frame] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    offset: Optional[Offset] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    staff: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    type: Optional[HarmonyType] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    print_frame: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-frame",
            "type": "Attribute",
        }
    )
    arrangement: Optional[HarmonyArrangement] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    system: Optional[SystemRelation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Metronome:
    """The metronome type represents metronome marks and other metric
    relationships.

    The beat-unit group and per-minute element specify regular metronome
    marks. The metronome-note and metronome-relation elements allow for
    the specification of metric modulations and other metric
    relationships, such as swing tempo marks where two eighths are
    equated to a quarter note / eighth note triplet. Tied notes can be
    represented in both types of metronome marks by using the beat-unit-
    tied and metronome-tied elements. The parentheses attribute
    indicates whether or not to put the metronome mark in parentheses;
    its value is no if not specified. The print-object attribute is set
    to no in cases where the metronome element represents a relationship
    or range that is not displayed in the music notation.

    :ivar beat_unit: The beat-unit element indicates the graphical note
        type to use in a metronome mark.
    :ivar beat_unit_dot: The beat-unit-dot element is used to specify
        any augmentation dots for a metronome mark note.
    :ivar beat_unit_tied:
    :ivar per_minute:
    :ivar metronome_arrows: If the metronome-arrows element is present,
        it indicates that metric modulation arrows are displayed on both
        sides of the metronome mark.
    :ivar metronome_note:
    :ivar metronome_relation: The metronome-relation element describes
        the relationship symbol that goes between the two sets of
        metronome-note elements. The currently allowed value is equals,
        but this may expand in future versions. If the element is empty,
        the equals value is used.
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar halign:
    :ivar valign:
    :ivar print_object:
    :ivar justify:
    :ivar parentheses:
    :ivar id:
    """
    class Meta:
        name = "metronome"

    beat_unit: List[NoteTypeValue] = field(
        default_factory=list,
        metadata={
            "name": "beat-unit",
            "type": "Element",
            "max_occurs": 2,
        }
    )
    beat_unit_dot: List[Empty] = field(
        default_factory=list,
        metadata={
            "name": "beat-unit-dot",
            "type": "Element",
        }
    )
    beat_unit_tied: List[BeatUnitTied] = field(
        default_factory=list,
        metadata={
            "name": "beat-unit-tied",
            "type": "Element",
        }
    )
    per_minute: Optional[PerMinute] = field(
        default=None,
        metadata={
            "name": "per-minute",
            "type": "Element",
        }
    )
    metronome_arrows: Optional[Empty] = field(
        default=None,
        metadata={
            "name": "metronome-arrows",
            "type": "Element",
        }
    )
    metronome_note: List[MetronomeNote] = field(
        default_factory=list,
        metadata={
            "name": "metronome-note",
            "type": "Element",
        }
    )
    metronome_relation: Optional[str] = field(
        default=None,
        metadata={
            "name": "metronome-relation",
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    halign: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    valign: Optional[Valign] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    justify: Optional[LeftCenterRight] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    parentheses: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Notations:
    """Notations refer to musical notations, not XML notations.

    Multiple notations are allowed in order to represent multiple
    editorial levels. The print-object attribute, added in Version 3.0,
    allows notations to represent details of performance technique, such
    as fingerings, without having them appear in the score.
    """
    class Meta:
        name = "notations"

    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    tied: List[Tied] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    slur: List[Slur] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    tuplet: List[Tuplet] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    glissando: List[Glissando] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    slide: List[Slide] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    ornaments: List[Ornaments] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    technical: List[Technical] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    articulations: List[Articulations] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    dynamics: List[Dynamics] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    fermata: List[Fermata] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    arpeggiate: List[Arpeggiate] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "sequential": True,
        }
    )
    non_arpeggiate: List[NonArpeggiate] = field(
        default_factory=list,
        metadata={
            "name": "non-arpeggiate",
            "type": "Element",
            "sequential": True,
        }
    )
    accidental_mark: List[AccidentalMark] = field(
        default_factory=list,
        metadata={
            "name": "accidental-mark",
            "type": "Element",
            "sequential": True,
        }
    )
    other_notation: List[OtherNotation] = field(
        default_factory=list,
        metadata={
            "name": "other-notation",
            "type": "Element",
            "sequential": True,
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Print:
    """The print type contains general printing parameters, including layout
    elements.

    The part-name-display and part-abbreviation-display elements may
    also be used here to change how a part name or abbreviation is
    displayed over the course of a piece. They take effect when the
    current measure or a succeeding measure starts a new system. Layout
    group elements in a print element only apply to the current page,
    system, or staff. Music that follows continues to take the default
    values from the layout determined by the defaults element.
    """
    class Meta:
        name = "print"

    page_layout: Optional[PageLayout] = field(
        default=None,
        metadata={
            "name": "page-layout",
            "type": "Element",
        }
    )
    system_layout: Optional[SystemLayout] = field(
        default=None,
        metadata={
            "name": "system-layout",
            "type": "Element",
        }
    )
    staff_layout: List[StaffLayout] = field(
        default_factory=list,
        metadata={
            "name": "staff-layout",
            "type": "Element",
        }
    )
    measure_layout: Optional[MeasureLayout] = field(
        default=None,
        metadata={
            "name": "measure-layout",
            "type": "Element",
        }
    )
    measure_numbering: Optional[MeasureNumbering] = field(
        default=None,
        metadata={
            "name": "measure-numbering",
            "type": "Element",
        }
    )
    part_name_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "part-name-display",
            "type": "Element",
        }
    )
    part_abbreviation_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "part-abbreviation-display",
            "type": "Element",
        }
    )
    staff_spacing: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "staff-spacing",
            "type": "Attribute",
        }
    )
    new_system: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "new-system",
            "type": "Attribute",
        }
    )
    new_page: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "new-page",
            "type": "Attribute",
        }
    )
    blank_page: Optional[int] = field(
        default=None,
        metadata={
            "name": "blank-page",
            "type": "Attribute",
        }
    )
    page_number: Optional[str] = field(
        default=None,
        metadata={
            "name": "page-number",
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class ScorePart:
    """The score-part type collects part-wide information for each part in a
    score.

    Often, each MusicXML part corresponds to a track in a Standard MIDI
    Format 1 file. In this case, the midi-device element is used to make
    a MIDI device or port assignment for the given track or specific
    MIDI instruments. Initial midi-instrument assignments may be made
    here as well. The score-instrument elements are used when there are
    multiple instruments per track.

    :ivar identification:
    :ivar part_link:
    :ivar part_name:
    :ivar part_name_display:
    :ivar part_abbreviation:
    :ivar part_abbreviation_display:
    :ivar group: The group element allows the use of different versions
        of the part for different purposes. Typical values include
        score, parts, sound, and data. Ordering information can be
        derived from the ordering within a MusicXML score or opus.
    :ivar score_instrument:
    :ivar player:
    :ivar midi_device:
    :ivar midi_instrument:
    :ivar id:
    """
    class Meta:
        name = "score-part"

    identification: Optional[Identification] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    part_link: List[PartLink] = field(
        default_factory=list,
        metadata={
            "name": "part-link",
            "type": "Element",
        }
    )
    part_name: Optional[PartName] = field(
        default=None,
        metadata={
            "name": "part-name",
            "type": "Element",
            "required": True,
        }
    )
    part_name_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "part-name-display",
            "type": "Element",
        }
    )
    part_abbreviation: Optional[PartName] = field(
        default=None,
        metadata={
            "name": "part-abbreviation",
            "type": "Element",
        }
    )
    part_abbreviation_display: Optional[NameDisplay] = field(
        default=None,
        metadata={
            "name": "part-abbreviation-display",
            "type": "Element",
        }
    )
    group: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    score_instrument: List[ScoreInstrument] = field(
        default_factory=list,
        metadata={
            "name": "score-instrument",
            "type": "Element",
        }
    )
    player: List[Player] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    midi_device: List[MidiDevice] = field(
        default_factory=list,
        metadata={
            "name": "midi-device",
            "type": "Element",
            "sequential": True,
        }
    )
    midi_instrument: List[MidiInstrument] = field(
        default_factory=list,
        metadata={
            "name": "midi-instrument",
            "type": "Element",
            "sequential": True,
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )


@dataclass
class DirectionType:
    """Textual direction types may have more than 1 component due to multiple
    fonts.

    The dynamics element may also be used in the notations element.
    Attribute groups related to print suggestions apply to the
    individual direction-type, not to the overall direction.

    :ivar rehearsal: The rehearsal element specifies letters, numbers,
        and section names that are notated in the score for reference
        during rehearsal. The enclosure is square if not specified. The
        language is Italian ("it") if not specified. Left justification
        is used if not specified.
    :ivar segno:
    :ivar coda:
    :ivar words: The words element specifies a standard text direction.
        The enclosure is none if not specified. The language is Italian
        ("it") if not specified. Left justification is used if not
        specified.
    :ivar symbol: The symbol element specifies a musical symbol using a
        canonical SMuFL glyph name. It is used when an occasional
        musical symbol is interspersed into text. It should not be used
        in place of semantic markup, such as metronome marks that mix
        text and symbols. Left justification is used if not specified.
        Enclosure is none if not specified.
    :ivar wedge:
    :ivar dynamics:
    :ivar dashes:
    :ivar bracket:
    :ivar pedal:
    :ivar metronome:
    :ivar octave_shift:
    :ivar harp_pedals:
    :ivar damp: The damp element specifies a harp damping mark.
    :ivar damp_all: The damp-all element specifies a harp damping mark
        for all strings.
    :ivar eyeglasses: The eyeglasses element represents the eyeglasses
        symbol, common in commercial music.
    :ivar string_mute:
    :ivar scordatura:
    :ivar image:
    :ivar principal_voice:
    :ivar percussion:
    :ivar accordion_registration:
    :ivar staff_divide:
    :ivar other_direction:
    :ivar id:
    """
    class Meta:
        name = "direction-type"

    rehearsal: List[FormattedTextId] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    segno: List[Segno] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    coda: List[Coda] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    words: List[FormattedTextId] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    symbol: List[FormattedSymbolId] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    wedge: Optional[Wedge] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    dynamics: List[Dynamics] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    dashes: Optional[Dashes] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    bracket: Optional[Bracket] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    pedal: Optional[Pedal] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    metronome: Optional[Metronome] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    octave_shift: Optional[OctaveShift] = field(
        default=None,
        metadata={
            "name": "octave-shift",
            "type": "Element",
        }
    )
    harp_pedals: Optional[HarpPedals] = field(
        default=None,
        metadata={
            "name": "harp-pedals",
            "type": "Element",
        }
    )
    damp: Optional[EmptyPrintStyleAlignId] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    damp_all: Optional[EmptyPrintStyleAlignId] = field(
        default=None,
        metadata={
            "name": "damp-all",
            "type": "Element",
        }
    )
    eyeglasses: Optional[EmptyPrintStyleAlignId] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    string_mute: Optional[StringMute] = field(
        default=None,
        metadata={
            "name": "string-mute",
            "type": "Element",
        }
    )
    scordatura: Optional[Scordatura] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    image: Optional[Image] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    principal_voice: Optional[PrincipalVoice] = field(
        default=None,
        metadata={
            "name": "principal-voice",
            "type": "Element",
        }
    )
    percussion: List[Percussion] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    accordion_registration: Optional[AccordionRegistration] = field(
        default=None,
        metadata={
            "name": "accordion-registration",
            "type": "Element",
        }
    )
    staff_divide: Optional[StaffDivide] = field(
        default=None,
        metadata={
            "name": "staff-divide",
            "type": "Element",
        }
    )
    other_direction: Optional[OtherDirection] = field(
        default=None,
        metadata={
            "name": "other-direction",
            "type": "Element",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class Note:
    """Notes are the most common type of MusicXML data. The MusicXML format
    distinguishes between elements used for sound information and elements used
    for notation information (e.g., tie is used for sound, tied for notation).
    Thus grace notes do not have a duration element. Cue notes have a duration
    element, as do forward elements, but no tie elements. Having these two
    types of information available can make interchange easier, as some
    programs handle one type of information more readily than the other. The
    print-leger attribute is used to indicate whether leger lines are printed.
    Notes without leger lines are used to indicate indeterminate high and low
    notes. By default, it is set to yes. If print-object is set to no, print-
    leger is interpreted to also be set to no if not present. This attribute is
    ignored for rests. The dynamics and end-dynamics attributes correspond to
    MIDI 1.0's Note On and Note Off velocities, respectively. They are
    expressed in terms of percentages of the default forte value (90 for MIDI
    1.0).

    The attack and release attributes are used to alter the starting and stopping time of the note from when it would otherwise occur based on the flow of durations - information that is specific to a performance. They are expressed in terms of divisions, either positive or negative. A note that starts a tie should not have a release attribute, and a note that stops a tie should not have an attack attribute. The attack and release attributes are independent of each other. The attack attribute only changes the starting time of a note, and the release attribute only changes the stopping time of a note.
    If a note is played only particular times through a repeat, the time-only attribute shows which times to play the note.
    The pizzicato attribute is used when just this note is sounded pizzicato, vs. the pizzicato element which changes overall playback between pizzicato and arco.

    :ivar grace:
    :ivar chord: The chord element indicates that this note is an
        additional chord tone with the preceding note. The duration of a
        chord note does not move the musical position within a measure.
        That is done by the duration of the first preceding note without
        a chord element. Thus the duration of a chord note cannot be
        longer than the preceding note. In most cases the duration will
        be the same as the preceding note. However it can be shorter in
        situations such as multiple stops for string instruments.
    :ivar pitch:
    :ivar unpitched:
    :ivar rest:
    :ivar tie:
    :ivar cue: The cue element indicates the presence of a cue note. In
        MusicXML, a cue note is a silent note with no playback. Normal
        notes that play can be specified as cue size using the type
        element. A cue note that is specified as full size using the
        type element will still remain silent.
    :ivar duration: Duration is a positive number specified in division
        units. This is the intended duration vs. notated duration (for
        instance, differences in dotted notes in Baroque-era music).
        Differences in duration specific to an interpretation or
        performance should be represented using the note element's
        attack and release attributes. The duration element moves the
        musical position when used in backup elements, forward elements,
        and note elements that do not contain a chord child element.
    :ivar instrument:
    :ivar footnote:
    :ivar level:
    :ivar voice:
    :ivar type:
    :ivar dot: One dot element is used for each dot of prolongation. The
        placement attribute is used to specify whether the dot should
        appear above or below the staff line. It is ignored for notes
        that appear on a staff space.
    :ivar accidental:
    :ivar time_modification:
    :ivar stem:
    :ivar notehead:
    :ivar notehead_text:
    :ivar staff: Staff assignment is only needed for music notated on
        multiple staves. Used by both notes and directions. Staff values
        are numbers, with 1 referring to the top-most staff in a part.
    :ivar beam:
    :ivar notations:
    :ivar lyric:
    :ivar play:
    :ivar listen:
    :ivar default_x:
    :ivar default_y:
    :ivar relative_x:
    :ivar relative_y:
    :ivar font_family:
    :ivar font_style:
    :ivar font_size:
    :ivar font_weight:
    :ivar color:
    :ivar print_object:
    :ivar print_dot:
    :ivar print_spacing:
    :ivar print_lyric:
    :ivar print_leger:
    :ivar dynamics:
    :ivar end_dynamics:
    :ivar attack:
    :ivar release:
    :ivar time_only:
    :ivar pizzicato:
    :ivar id:
    """
    class Meta:
        name = "note"

    grace: Optional[Grace] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    chord: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 4,
            "sequential": True,
        }
    )
    pitch: List[Pitch] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 4,
            "sequential": True,
        }
    )
    unpitched: List[Unpitched] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 4,
            "sequential": True,
        }
    )
    rest: List[Rest] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 4,
            "sequential": True,
        }
    )
    tie: List[Tie] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 4,
        }
    )
    cue: List[Empty] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 2,
            "sequential": True,
        }
    )
    duration: List[Decimal] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 2,
            "min_exclusive": Decimal("0"),
            "sequential": True,
        }
    )
    instrument: List[Instrument] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    voice: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    type: Optional[NoteType] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    dot: List[EmptyPlacement] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    accidental: Optional[Accidental] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    time_modification: Optional[TimeModification] = field(
        default=None,
        metadata={
            "name": "time-modification",
            "type": "Element",
        }
    )
    stem: Optional[Stem] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    notehead: Optional[Notehead] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    notehead_text: Optional[NoteheadText] = field(
        default=None,
        metadata={
            "name": "notehead-text",
            "type": "Element",
        }
    )
    staff: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    beam: List[Beam] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "max_occurs": 8,
        }
    )
    notations: List[Notations] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    lyric: List[Lyric] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    play: Optional[Play] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    listen: Optional[Listen] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    default_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-x",
            "type": "Attribute",
        }
    )
    default_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "default-y",
            "type": "Attribute",
        }
    )
    relative_x: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-x",
            "type": "Attribute",
        }
    )
    relative_y: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "relative-y",
            "type": "Attribute",
        }
    )
    font_family: Optional[str] = field(
        default=None,
        metadata={
            "name": "font-family",
            "type": "Attribute",
            "pattern": r"[^,]+(, ?[^,]+)*",
        }
    )
    font_style: Optional[FontStyle] = field(
        default=None,
        metadata={
            "name": "font-style",
            "type": "Attribute",
        }
    )
    font_size: Optional[Union[Decimal, CssFontSize]] = field(
        default=None,
        metadata={
            "name": "font-size",
            "type": "Attribute",
        }
    )
    font_weight: Optional[FontWeight] = field(
        default=None,
        metadata={
            "name": "font-weight",
            "type": "Attribute",
        }
    )
    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "pattern": r"#[\dA-F]{6}([\dA-F][\dA-F])?",
        }
    )
    print_object: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-object",
            "type": "Attribute",
        }
    )
    print_dot: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-dot",
            "type": "Attribute",
        }
    )
    print_spacing: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-spacing",
            "type": "Attribute",
        }
    )
    print_lyric: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-lyric",
            "type": "Attribute",
        }
    )
    print_leger: Optional[YesNo] = field(
        default=None,
        metadata={
            "name": "print-leger",
            "type": "Attribute",
        }
    )
    dynamics: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
        }
    )
    end_dynamics: Optional[Decimal] = field(
        default=None,
        metadata={
            "name": "end-dynamics",
            "type": "Attribute",
            "min_inclusive": Decimal("0"),
        }
    )
    attack: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    release: Optional[Decimal] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    time_only: Optional[str] = field(
        default=None,
        metadata={
            "name": "time-only",
            "type": "Attribute",
            "pattern": r"[1-9][0-9]*(, ?[1-9][0-9]*)*",
        }
    )
    pizzicato: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class PartList:
    """The part-list identifies the different musical parts in this document.

    Each part has an ID that is used later within the musical data.
    Since parts may be encoded separately and combined later,
    identification elements are present at both the score and score-part
    levels. There must be at least one score-part, combined as desired
    with part-group elements that indicate braces and brackets. Parts
    are ordered from top to bottom in a score based on the order in
    which they appear in the part-list.

    :ivar part_group:
    :ivar score_part: Each MusicXML part corresponds to a track in a
        Standard MIDI Format 1 file. The score-instrument elements are
        used when there are multiple instruments per track. The midi-
        device element is used to make a MIDI device or port assignment
        for the given track. Initial midi-instrument assignments may be
        made here as well.
    """
    class Meta:
        name = "part-list"

    part_group: List[PartGroup] = field(
        default_factory=list,
        metadata={
            "name": "part-group",
            "type": "Element",
            "sequential": True,
        }
    )
    score_part: List[ScorePart] = field(
        default_factory=list,
        metadata={
            "name": "score-part",
            "type": "Element",
            "sequential": True,
        }
    )


@dataclass
class Direction:
    """A direction is a musical indication that is not necessarily attached to
    a specific note.

    Two or more may be combined to indicate words followed by the start
    of a dashed line, the end of a wedge followed by dynamics, etc. For
    applications where a specific direction is indeed attached to a
    specific note, the direction element can be associated with the
    first note element that follows it in score order that is not in a
    different voice. By default, a series of direction-type elements and
    a series of child elements of a direction-type within a single
    direction element follow one another in sequence visually. For a
    series of direction-type children, non-positional formatting
    attributes are carried over from the previous element by default.

    :ivar direction_type:
    :ivar offset:
    :ivar footnote:
    :ivar level:
    :ivar voice:
    :ivar staff: Staff assignment is only needed for music notated on
        multiple staves. Used by both notes and directions. Staff values
        are numbers, with 1 referring to the top-most staff in a part.
    :ivar sound:
    :ivar listening:
    :ivar placement:
    :ivar directive:
    :ivar system:
    :ivar id:
    """
    class Meta:
        name = "direction"

    direction_type: List[DirectionType] = field(
        default_factory=list,
        metadata={
            "name": "direction-type",
            "type": "Element",
            "min_occurs": 1,
        }
    )
    offset: Optional[Offset] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    footnote: Optional[FormattedText] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    level: Optional[Level] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    voice: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    staff: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    sound: Optional[Sound] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    listening: Optional[Listening] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    placement: Optional[AboveBelow] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    directive: Optional[YesNo] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    system: Optional[SystemRelation] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )
    id: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
        }
    )


@dataclass
class ScorePartwise:
    """The score-partwise element is the root element for a partwise MusicXML
    score.

    It includes a score-header group followed by a series of parts with
    measures inside. The document-attributes attribute group includes
    the version attribute.

    :ivar work:
    :ivar movement_number: The movement-number element specifies the
        number of a movement.
    :ivar movement_title: The movement-title element specifies the title
        of a movement, not including its number.
    :ivar identification:
    :ivar defaults:
    :ivar credit:
    :ivar part_list:
    :ivar part:
    :ivar version:
    """
    class Meta:
        name = "score-partwise"

    work: Optional[Work] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    movement_number: Optional[str] = field(
        default=None,
        metadata={
            "name": "movement-number",
            "type": "Element",
        }
    )
    movement_title: Optional[str] = field(
        default=None,
        metadata={
            "name": "movement-title",
            "type": "Element",
        }
    )
    identification: Optional[Identification] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    defaults: Optional[Defaults] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    credit: List[Credit] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    part_list: Optional[PartList] = field(
        default=None,
        metadata={
            "name": "part-list",
            "type": "Element",
            "required": True,
        }
    )
    part: List["ScorePartwise.Part"] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )
    version: str = field(
        default="1.0",
        metadata={
            "type": "Attribute",
        }
    )

    @dataclass
    class Part:
        measure: List["ScorePartwise.Part.Measure"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "min_occurs": 1,
            }
        )
        id: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )

        @dataclass
        class Measure:
            note: List[Note] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            backup: List[Backup] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            forward: List[Forward] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            direction: List[Direction] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            attributes: List[Attributes] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            harmony: List[Harmony] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            figured_bass: List[FiguredBass] = field(
                default_factory=list,
                metadata={
                    "name": "figured-bass",
                    "type": "Element",
                    "sequential": True,
                }
            )
            print: List[Print] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            sound: List[Sound] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            listening: List[Listening] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            barline: List[Barline] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            grouping: List[Grouping] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            link: List[Link] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            bookmark: List[Bookmark] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            number: Optional[str] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                }
            )
            text: Optional[str] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "min_length": 1,
                }
            )
            implicit: Optional[YesNo] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                }
            )
            non_controlling: Optional[YesNo] = field(
                default=None,
                metadata={
                    "name": "non-controlling",
                    "type": "Attribute",
                }
            )
            width: Optional[Decimal] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                }
            )
            id: Optional[str] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                }
            )


@dataclass
class ScoreTimewise:
    """The score-timewise element is the root element for a timewise MusicXML
    score.

    It includes a score-header group followed by a series of measures
    with parts inside. The document-attributes attribute group includes
    the version attribute.

    :ivar work:
    :ivar movement_number: The movement-number element specifies the
        number of a movement.
    :ivar movement_title: The movement-title element specifies the title
        of a movement, not including its number.
    :ivar identification:
    :ivar defaults:
    :ivar credit:
    :ivar part_list:
    :ivar measure:
    :ivar version:
    """
    class Meta:
        name = "score-timewise"

    work: Optional[Work] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    movement_number: Optional[str] = field(
        default=None,
        metadata={
            "name": "movement-number",
            "type": "Element",
        }
    )
    movement_title: Optional[str] = field(
        default=None,
        metadata={
            "name": "movement-title",
            "type": "Element",
        }
    )
    identification: Optional[Identification] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    defaults: Optional[Defaults] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    credit: List[Credit] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )
    part_list: Optional[PartList] = field(
        default=None,
        metadata={
            "name": "part-list",
            "type": "Element",
            "required": True,
        }
    )
    measure: List["ScoreTimewise.Measure"] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )
    version: str = field(
        default="1.0",
        metadata={
            "type": "Attribute",
        }
    )

    @dataclass
    class Measure:
        part: List["ScoreTimewise.Measure.Part"] = field(
            default_factory=list,
            metadata={
                "type": "Element",
                "min_occurs": 1,
            }
        )
        number: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "required": True,
            }
        )
        text: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
                "min_length": 1,
            }
        )
        implicit: Optional[YesNo] = field(
            default=None,
            metadata={
                "type": "Attribute",
            }
        )
        non_controlling: Optional[YesNo] = field(
            default=None,
            metadata={
                "name": "non-controlling",
                "type": "Attribute",
            }
        )
        width: Optional[Decimal] = field(
            default=None,
            metadata={
                "type": "Attribute",
            }
        )
        id: Optional[str] = field(
            default=None,
            metadata={
                "type": "Attribute",
            }
        )

        @dataclass
        class Part:
            note: List[Note] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            backup: List[Backup] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            forward: List[Forward] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            direction: List[Direction] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            attributes: List[Attributes] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            harmony: List[Harmony] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            figured_bass: List[FiguredBass] = field(
                default_factory=list,
                metadata={
                    "name": "figured-bass",
                    "type": "Element",
                    "sequential": True,
                }
            )
            print: List[Print] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            sound: List[Sound] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            listening: List[Listening] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            barline: List[Barline] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            grouping: List[Grouping] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            link: List[Link] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            bookmark: List[Bookmark] = field(
                default_factory=list,
                metadata={
                    "type": "Element",
                    "sequential": True,
                }
            )
            id: Optional[str] = field(
                default=None,
                metadata={
                    "type": "Attribute",
                    "required": True,
                }
            )
