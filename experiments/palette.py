"""One colour vocabulary for every figure in the paper.

Okabe-Ito hues, so no red/green pair ever carries meaning on its own, and every
fill is paired with a border dark enough that the figures still read when the
proceedings print in greyscale.

Two ideas, used consistently:

    REGIONS carry a tint, and the blocks inside them stay white.  Where a thing
    sits says what it is, so the architecture figure needs almost no arrows to
    explain itself:

        HOST    purple     the processor side
        LOGIC   green      the programmable logic
        LANES   green, a shade deeper -- the replicated part, nested inside
        CACHE   blue       memory, and the KV cache in it

    FLOW colours separate the two directions of the loop, in the figures and
    the plots alike:

        WRITE   vermilion  a token going into the cache
        READ    blue       the scan coming back out

The plots reuse READ for what this design does and what wins, WRITE for the
alternative being argued against, and grey for neutral.
"""

# regions
HOST_FILL,  HOST_EDGE  = "#EDE4F3", "#7B5EA7"
LOGIC_FILL, LOGIC_EDGE = "#E6F0E2", "#4C8C3F"
LANES_FILL, LANES_EDGE = "#D4E7CE", "#4C8C3F"
CACHE_FILL, CACHE_EDGE = "#D9E7F5", "#0072B2"
CELL_FILL              = "#AECDE9"

# flow
WRITE = "#D55E00"
READ  = "#0072B2"

# plots
BLUE       = READ
BLUE_FILL  = "#CFE2F3"
ORANGE     = "#E69F00"
ORANGE_FILL = "#FBE6C2"
GREY       = "#8A8A8A"
GREY_FILL  = "#E4E4E4"
VERMILION  = WRITE
INK        = "#1A1A1A"
