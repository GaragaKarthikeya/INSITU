"""One colour vocabulary for every figure in the paper.

The hues are Okabe-Ito, which stays legible under the common forms of colour
blindness -- no red/green pair carries meaning anywhere -- and the two primaries
are far enough apart in lightness that the figures still separate when the
proceedings print in greyscale.

What each colour means, held constant across all three figures so a reader who
learns it once does not have to relearn it:

    BLUE        what this design does, and what wins
                -- the compressed row in the architecture figure
                -- the Pareto frontier, and the configurations the board can
                   actually build, in the footprint plot
                -- more key bits than value bits in the asymmetry plot

    ORANGE      the alternative being argued against
                -- more value bits than key bits

    GREY        neutral: equal split, or a configuration that does not fit

    VERMILION   used once, for 4b/2b -- the configuration that shipped
"""

BLUE = "#0072B2"
BLUE_FILL = "#CFE2F3"
ORANGE = "#E69F00"
ORANGE_FILL = "#FBE6C2"
GREY = "#8A8A8A"
GREY_FILL = "#E4E4E4"
VERMILION = "#D55E00"
INK = "#1A1A1A"
