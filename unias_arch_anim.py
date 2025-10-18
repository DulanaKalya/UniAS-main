from manim import *

class UniASArchitecture(Scene):
    def construct(self):
        # Input image block (with fallback rectangle if missing)
        try:
            input_img = ImageMobject("img/input.png").scale(0.8)
        except:
            input_img = Rectangle(width=2.2, height=2.2, color=GREY).set_fill(GREY_E, opacity=0.5)

        input_group = Group(
            input_img,
            Text("Input Image", font_size=28).next_to(input_img, DOWN, buff=0.2)
        ).to_edge(LEFT).shift(UP*1.5)

        # Backbone (EfficientNet)
        backbone = Rectangle(width=3, height=1.5, color=BLUE).set_fill(BLUE_E, opacity=0.5)
        backbone_label = Text("EfficientNet-B4", font_size=28).move_to(backbone.get_center())
        backbone_group = Group(backbone, backbone_label).next_to(input_group, RIGHT, buff=1)

        # Transformer Encoder
        transformer = Rectangle(width=3.5, height=1.5, color=GREEN).set_fill(GREEN_E, opacity=0.5)
        transformer_label = Text("Transformer Encoder", font_size=28).move_to(transformer.get_center())
        transformer_group = Group(transformer, transformer_label).next_to(backbone_group, RIGHT, buff=1)

        # Output (Anomaly Map)
        output = Rectangle(width=2.5, height=1.5, color=RED).set_fill(RED_E, opacity=0.5)
        output_label = Text("Anomaly Map", font_size=28).move_to(output.get_center())
        output_group = Group(output, output_label).next_to(transformer_group, RIGHT, buff=1)

        # Arrows
        arrow1 = Arrow(input_group.get_right(), backbone_group.get_left(), buff=0.1)
        arrow2 = Arrow(backbone_group.get_right(), transformer_group.get_left(), buff=0.1)
        arrow3 = Arrow(transformer_group.get_right(), output_group.get_left(), buff=0.1)

        # Animate step by step
        self.play(FadeIn(input_group))
        self.play(GrowArrow(arrow1), FadeIn(backbone_group))
        self.play(GrowArrow(arrow2), FadeIn(transformer_group))
        self.play(GrowArrow(arrow3), FadeIn(output_group))
        self.wait(2)
