"""Structured prompts used by the Judge bootstrap, the online Judge and the exporters.

The required output contract comes from :mod:`rfs.judge_training.judge_schema` so the
prompt, the dataset validator, and the evaluator can never disagree about fields.

The online repair Judge and the training exporters share this file on purpose: a Judge
trained on one prompt and served another has not been trained at all. The repair schema
below is therefore the pipeline's own ``rfs.judge_result`` contract, so the model answers
in the vocabulary the controller already applies.

One loop needs two different answers, so it asks two different questions and keeps them
apart. The advisory role names the defect and the bounded repair it wants, and never
scores: the score it used to emit was read as a verdict it had not been asked to give.
The acceptance role scores the figure and names what blocks acceptance, and never
proposes a repair. ``judge_role_family`` is the only place that decides which of the two
questions a served role is answering.
"""

from __future__ import annotations

import json
from typing import Any

from ..contracts.evolution import ADVISORY_ROLE_FAMILY, ACCEPTANCE_ROLE_FAMILY, REPAIR_ACTIONS, SCOPE_TYPES, STAGE_ORDER, judge_role_family
from ..contracts.fault_attribution import render_fault_attribution_for_prompt
from ..contracts.repair_actions import render_action_catalog_for_prompt
from ..planning.prompt_assembly import render_patch_block
from .judge_schema import OBSERVED_FIELDS, PAIRWISE_TASK_TYPES, PROCESS_TASK_TYPES, judgement_schema


# ``judge-online-v010`` requires the reflection to fill the rule field whenever a round
# taught something about how figures are drawn, and says that stating a rule is not a claim
# that this round went wrong. v009 asked the advisory answer to name whether the repair it
# proposes should be examined as common or only in this paper's domain. v008 inserted the
# bounded-action catalog. Changing the served text without a version bump would train
# and serve two different questions.
# ``judge-online-v013`` asks the advisory role to name the stage a defect entered at, and to
# back that with a field the round's stage trace actually measured. The vocabulary it may name
# is the contract's own stage order, so the answer can be checked rather than believed.
# ``judge-online-v014`` adds the two mechanism-level rules to the advisory menu and the two
# fields that carry their parameters. It is the same question with more ways to answer: a
# defect that is really about how such figures are drawn can now be asked for as a rule, so
# the loop stops being offered only moves on the figure in front of it.
# ``judge-online-v016`` widens the vocabulary a fault attribution may cite. The seat was told to
# name a key of ``process_context.stage_trace.metrics`` alone, and it kept citing the other
# evidence blocks it had been shown -- the colour report's own feedback entries, the grounding
# Gate's coverages -- which the gate then discarded as unreadable. Every attribution of a real
# defect was thrown away with them.
# ``judge-online-v018`` gives ``follow_up`` the sentence it was always meant to answer and
# says what a move on the figure is answered against. The controller has accepted
# ``follow_up`` since ``REFLECTION_POLICIES`` was written, and the executor, the three-attempt
# budget and the write-off path were all in place, but the menu never offered it -- so the
# seat answered ``keep``/``narrow_scope``/``revise_rule`` every time (0 of 257 reflections
# chose it) and the answer it did write, ``narrow_scope``, means the opposite: that the
# experience should stop being offered, not that the lesson deserves one more attempt. The
# same edit says what a round that moved the figure is given to check an attribution
# against, because the two attributions that earn an attempt used to be refused for every
# round that ran no rule.
# ``judge-online-v017`` stops the reflection presupposing a repair request. Round 0 of an
# iteration that inherited a finished figure draws the run's own figure, and a round that
# drew it was still asked "was the repair effective" -- a question about a request it never
# received. The round now says what it taught either way, which is what lets the winning
# round state the rule it just demonstrated instead of writing nothing.
# ``judge-online-v011`` lets the advisory role be asked the question the Judge was actually
# trained on. It was trained on two figures at a time: shown one figure and asked what is wrong
# with it, it answers "no defect" whatever the figure looks like, not because the figure is good
# but because that is not the question it learned. A row that carries the version the loop has
# already accepted may now travel with it, and the advisory schema grows the two fields that
# question needs. Every other row -- and both measurement roles -- renders exactly as v010.
# ``judge-online-v020`` writes down when each action is the answer. v014 put the two mechanism
# rules on the menu and v008 put the catalog in the prompt, but a row only said what an action
# does and what it cannot. A seat that knows an action exists but not what a fault looks like
# when that is the action to reach for answers with the name it is surest of, which is how a
# menu of twelve repairs kept coming back as the same one. Every row now carries ``when``, the
# menu renders grouped by the layer that answers the fault -- what this round moves itself, the
# rules the compile applies to later figures, the requests that go upstream -- and the prompt no
# longer prints the shipped default budgets, because a seat that reads those numbers back is
# naming the policy experience has to differ from. The question is unchanged; the menu is
# readable. (v019 is the comparison-verdict wording, written on another branch of this file; the
# number is not reused here so the two texts never share a version.)
# ``judge-online-v021`` tells the advisory role which requirement a stage failed to write, and
# hands it the table the answer is chosen from. v013 asked it to name the stage a defect entered
# at and v016 widened the fields it could cite, but the answer stopped at the stage: "the layout
# is bad" names nowhere a prompt can be changed. The seat is now given the symptom-to-stage table
# -- including which stage is the cheap one -- and asked for the requirement that went unwritten
# there, so the answer lands on a sentence a planning prompt could carry.
# ``judge-online-v022`` asks for the two fields the reflection was already told about. v021's text
# described ``prompt_patch`` and ``claims`` in its later paragraphs, but the sentence that lists
# the fields the answer must carry -- the one every model reads first and answers to -- stopped at
# ``rule``. The seat answered exactly that list: over a whole run of rounds it wrote a defect list,
# a why_changed and a mechanism rule, and left ``prompt_patch`` empty and ``claims`` a bare list
# every single time, so the two channels that carry a lesson to a later round had nothing to
# register, inject or settle. The list and the paragraphs name the same fields now.
# ``judge-online-v023`` says where the patch's sentence comes from. Naming the field was not
# enough: over the next two rounds the seat answered the same question in the one place it had
# already answered it -- it named the stage whose requirement went unwritten in its own
# ``fault_stage`` and ``fault_requirement``, and left ``prompt_patch`` an empty object. The
# paragraph below now says the patch is that same sentence said again for a later prompt, and
# that a mechanism rule is not a substitute for it.
# ``judge-online-v026`` says what a crossing is. The defect the gate reported for a connector
# running through a card was called ``interface:path_crossing`` even though the loop's own
# crossing count -- two connectors crossing each other -- was a different reading, so a lesson
# about one was filed against the other. The gate names the box pass for what it is now, and the
# menu of numbers quoted to the seat follows it.
# ``judge-online-v027`` asks the seat where it has finished. Every round was measured layer by
# layer and nothing ever recorded that a layer was done with, so the loop's only account of where
# it stood was "this round changed nothing", and a settled structure could be re-planned by one
# sentence on the next call. ``layer_settled`` is the seat's own answer and the loop records it;
# the readings stay evidence, and no reading is ever read as a verdict about a layer.
# ``judge-online-v028`` says in what order a figure is judged. The seat was reading one defect
# -- a connector crossing a panel title -- as "this whole figure is the worse one", which is how a
# well-planned, well-coloured version lost to a sparser one. Both questions that ask which of two
# figures is better now carry the same rule: the whole figure first, the parts after it, and clean
# lines do not buy back a worse structure.
# ``judge-online-v029`` shows the seat what the stage already asks for. It was told to name the
# requirement a stage failed to write and it wrote the same sentence every round, because the only
# requirement it could see was the one in front of it: the stage's own requirement list, the
# sentences already standing there and the sentences already in the library now travel with the
# round, and the patch that repeats one already filed is folded into it rather than filed again.
# ``judge-online-v025`` hands the seat a price list instead of a list of refusals. The advisory
# seat was shown which repair layers the executor would refuse, which was never a preference the
# seat could weigh -- it was a door, and the door is gone: nothing refuses a repair for the layer
# it works at any more, and what bounds a round is its budget and its clock. What the seat is
# shown now is what each layer spends -- seconds for a program edit, one compile for a rule,
# minutes to hours and a possible redraw for a stage re-run -- together with the reading the loop
# itself took of the figure, so the choice of how much to spend is one the seat can make and
# defend rather than one the mechanism had already made.
# ``judge-online-v030`` makes the transferable rule a requirement rather than a preference. A
# reflection could answer "the figure improved" and leave the rule empty, and the round then had
# nothing a later paper could be drawn by -- seventeen accepted rounds were refused for exactly
# that, and the library stayed at one real lesson. A winning round now has to either state the
# rule or say, in the same answer, what about it cannot be reused.
# ``judge-online-v031`` says in the reflection's own words why a layer that is already good is what
# moves the round down. The question about the round's layer was on the sheet and the field was in
# the answer, and the seat still read "is this layer finished with" as a verdict to withhold -- so
# the sentence now says outright that naming a layer good is what carries the round to the layer
# below it, and that leaving the field empty means the layer is not good yet.
PROMPT_VERSION = "judge-online-v031"

ADVISORY_IDENTITY = (
    "You are a strict scientific framework-figure reviewer. Read the paper facts and evidence, then inspect the supplied image(s). "
    "Scientific correctness and grounding take priority over aesthetics. Do not invent facts. "
    "Your only job is to name the defects this figure has and the bounded repair each one requires: you are not asked to score it and not asked whether it may be accepted. "
    "When you name a repair, also set applicability to common or domain. common means this lesson should be examined on papers in other domains as well as this one. "
    "domain means it should be examined only in this paper's own domain. Do not invent a domain name, and do not decide from how similar two figures look. "
    "If this paper has no domain, do not write domain. "
    "Every defect is written twice, in two different fields. problem is about this figure and may name this figure's own boxes. "
    "transferable_advice is the part another paper could be drawn by: one or two sentences on how a figure with this defect ought to be drawn, naming no node label, no panel title, no object id and no coordinate. "
    "Write the advice even when this figure's own repair is obvious; leave it empty only when the round taught nothing about how such figures are drawn."
)
ACCEPTANCE_IDENTITY = (
    "You are an independent scientific framework-figure acceptance judge. Read the paper facts and evidence, then inspect the supplied image(s). "
    "Scientific correctness and grounding take priority over aesthetics. Do not invent facts. "
    "Your only job is to score this figure and to name what blocks acceptance: you are not asked which repair to run, and a repair action does not belong in your answer."
)

ADVISORY_OUTPUT_SCHEMA: dict[str, Any] = {
    "status": "pass|fault|blocked",
    "applicability": "common|domain",
    "preserve": ["what must not change"],
    "blocking_issues": ["hard problem that must be repaired"],
    "fault_stage": "|".join(STAGE_ORDER) + "|\u2014 the stage this defect entered at, or empty when it entered nowhere in particular",
    "fault_requirement": "the one requirement that stage should have written and did not, as the symptom-to-stage table words it; empty only when fault_stage is empty",
    "fault_evidence": [
        {
            "stage": "the stage you named in fault_stage",
            "field": "a field this round put in front of you, spelled as that block spells it: a key of process_context.stage_trace.metrics such as panel_occupancy, or a field of another block you were shown such as grounding_gate.edge_coverage or color_report.feedback[0].reason",
            "value": "the number or line you read in that field",
            "bound": "the value it should have stayed within",
        }
    ],
    "stage_feedback": [
        {
            "scope_type": "|".join(sorted(SCOPE_TYPES)),
            "problem": "one sentence naming the defect",
            "transferable_advice": "one or two sentences on how a figure with this defect ought to be drawn; names no node label, object id or coordinate",
            "evidence": ["short quote, measurement or object id"],
            "semantic_node_ids": ["scientific node id"],
            "slot_ids": ["layout slot id"],
            "object_ids": ["editable object id"],
            "edge_ids": ["connector id"],
            "expected_effect": "what the repair must achieve",
            "forbidden_changes": ["what the repair must not touch"],
            "requested_actions": [
                {
                    "action": "|".join(sorted(REPAIR_ACTIONS)),
                    "factor": "unsigned scale about the box centre; required when the named action lists budget_keys",
                    "dx": "unsigned horizontal stagger; required when the named action lists budget_keys",
                    "dy": "unsigned vertical shift; required when the named action lists budget_keys",
                    "constraints": {
                        "shared_vertical_channel": "boolean; required when the action is router_constraint",
                        "forbid_crossing": "boolean; required when the action is router_constraint",
                        "min_bend": "smallest number of segments a route may use",
                        "max_segments": "largest number of segments a route may use",
                    },
                    "params": {
                        "align_grid": "grid step as a fraction of the canvas; required when the action is layout_constraint",
                        "gutter_min": "smallest gap between neighbouring boxes",
                        "panel_occupancy_min": "share of its panel a group of boxes must fill",
                    },
                }
            ],
        }
    ],
}

ACCEPTANCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "overall_score": "number 0..1",
    "dimension_scores": {"<dimension>": "number 0..1"},
    "status": "pass|fault|blocked",
    "confidence": "number 0..1",
    "blocking_issues": ["hard problem that blocks acceptance"],
}

#: The third question, and the one the loop decides on: it shows the accepted state and the
#: candidate together and asks which is better, rather than asking twice for an absolute number
#: and subtracting. The asked-for vocabulary is the training contract's own
#: (``preferred_candidate`` in :mod:`rfs.judge_training.judge_schema`), so an online answer is
#: already in the shape a preference pair can be supervised in.
PAIRWISE_TASK_TYPE = "pairwise"
PAIRWISE_ACCEPTANCE_IDENTITY = (
    "You are an independent scientific framework-figure acceptance judge comparing two versions of one figure. "
    "Read the paper facts and evidence, then inspect the two supplied images. Scientific correctness and grounding take "
    "priority over aesthetics. Do not invent facts. Your only job is to say which version a reader of this paper is better "
    "served by, how far apart they are, and what still blocks acceptance of the one you preferred: you are not asked to score "
    "either version and not asked which repair to run."
)
PAIRWISE_ACCEPTANCE_OUTPUT_SCHEMA: dict[str, Any] = {
    "preferred_candidate": "A|B|tie",
    "margin": "none|slight|clear",
    "confidence": "number 0..1",
    "blocking_issues": ["hard problem that still blocks acceptance of the version you preferred"],
}
PAIRWISE_MARGIN_RULE = (
    "margin describes how far apart the two versions are on the single thing that decides it: clear when one of them "
    "removes or introduces a defect a reader would notice, slight when one is better only in degree, none when you cannot "
    "tell them apart. A tie is a real answer and is preferred over guessing."
)

#: The order a comparison is judged in, and the one trade a person has already said out loud. A
#: figure is read as a whole before anything in it is read, and clean lines are not worth a worse
#: structure: a well-planned figure stays the better one while it carries a few crossed or stray
#: connectors, and the seat says what still needs repairing instead of swapping back to a version
#: whose structure is worse. Both questions that ask "which of these two is better" -- the advisory
#: comparison and the pairwise acceptance -- are told the same thing, so the two seats cannot answer
#: the same pair by two different rules.
WHOLE_FIRST_RULE = (
    "Judge the whole figure before you judge anything in it. Take A and B in as a reader of this "
    "paper would: how well each one carries the paper's argument, how legible its parts are, and how "
    "the parts sit together as one figure. Only then name the defects that separate them. A crossed "
    "or stray connector, a line running past a box or a small overlap does not by itself make a "
    "figure the worse one: when one version is better planned, better laid out or better coloured, "
    "prefer that version even while it carries a few such lines, and say what still needs repairing "
    "rather than swapping to a version whose structure is worse. Legible content that carries the "
    "argument outweighs clean lines, so a figure that fails only in cleanliness is preferred over "
    "one that fails in content, structure or readability."
)

REPAIR_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    ADVISORY_ROLE_FAMILY: ADVISORY_OUTPUT_SCHEMA,
    ACCEPTANCE_ROLE_FAMILY: ACCEPTANCE_OUTPUT_SCHEMA,
}

PROCESS_OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "outcome": {"state_after": "object", "repair_gain": "number", "grounding_gate": "object", "decision": "accept|rollback", "audit_status": "verified status"},
    "trajectory": {"state_before": "object", "diagnosis": "object", "action": "bounded action object", "state_after": "object", "object_diff": "object", "repair_gain": "number", "grounding_gate": "object", "decision": "accept|rollback", "reflection": "object", "audit_status": "verified status"},
}

#: The reflection may name a replacement action, so it is given the same action vocabulary the
#: controller applies. A name outside it is re-read deterministically downstream, and a name
#: outside even that is refused with an event rather than silently guessed at. It may also state
#: a mechanism-level rule, which is the part of a lesson that another paper can be drawn by;
#: that vocabulary is narrower, carries no ids, and is refused the same way when it is wrong.
REFLECTION_PROMPT = (
    "Reflect on what this round taught. Distinguish domain mismatch, "
    "excessive scope, incorrect attribution, global regression, and random generation variance. "
    "Return JSON only with outcome improved|regressed|inconclusive, defects, failure_type, "
    "why_changed, scope_assessment appropriate|too_broad|too_narrow, domain_mismatch, "
    "randomness_risk, measurement_noise_risk, next_policy "
    "keep|narrow_scope|revise|revise_rule|follow_up|disable, revised_action, binding, rule, "
    "prompt_patch, layer_settled, and claims. "
    "outcome says which way the figure moved and defects says what changed, so answer them as "
    "two statements: defects.cleared lists the problems this round fixed, defects.introduced "
    "lists the ones it created, and failure_type is only meaningful when the round did not "
    "improve -- write none beside an improvement, and never name a regression and an "
    "improvement in the same row. "
    "Two different doubts have two different fields. randomness_risk is about the drawing: the "
    "generation is stochastic, so this round's figure could have come out differently. "
    "measurement_noise_risk is about the verdict this round was decided on: comparing these two "
    "states again could come out the other way, so the decision itself is not trustworthy. Say "
    "which one you mean. A defect list you were shown that looks invented or unreachable is "
    "neither of the two, and belongs in why_changed. "
    "revised_action.type must be exactly one of "
    + "|".join(sorted(REPAIR_ACTIONS))
    + ", together with the scope_type it applies to; any other name is discarded. "
    "Split the lesson into binding and rule. binding names the objects of this figure the move "
    "touched, and no other paper has them. rule is the part that travels, and it is only worth "
    "writing when the round showed that a way of drawing -- not this figure's boxes -- would "
    "help: an object with type router_constraint and constraints drawn from "
    "shared_vertical_channel|forbid_crossing|min_bend|max_segments, or type layout_constraint "
    "and params drawn from align_grid|gutter_min|panel_occupancy_min. A rule carries no object "
    "ids and no coordinates; anything else is discarded. Written out, three rules that travel: "
    '{"type": "router_constraint", "constraints": {"shared_vertical_channel": true, '
    '"forbid_crossing": true}}, '
    '{"type": "layout_constraint", "params": {"align_grid": 0.05, "gutter_min": 0.02}} and '
    '{"type": "layout_constraint", "params": {"panel_occupancy_min": 0.5}}. '
    "next_policy revise_rule keeps that rule for later papers and leaves this round's own "
    "bounded action unchanged. "
    "narrow_scope and follow_up both answer a lesson that did not land, and they are not the "
    "same answer: narrow_scope says this experience should stop being offered under a "
    "condition like this one, so it will not be tried here again; follow_up says the lesson "
    "stands and this round is worth one more cheap attempt. Say follow_up only when you can "
    "also say which attempt you mean -- refined_rule, or follow_up -- because an attempt the "
    "loop cannot aim is refused rather than run. "
    "Always fill both fields. Set rule to {} only when this round taught nothing about how figures "
    "are drawn; whenever it did, rule must name one of the two mechanism actions with its "
    "parameters, and next_policy must be revise_rule. A rule a later paper is drawn by is worth "
    "keeping even when this round's own move was right, so stating one is not a claim that this "
    "round went wrong. "
    "This is a requirement, not a preference: a round that moved the figure forward and leaves rule "
    "empty must say so in the same answer -- one sentence in why_changed naming what about this "
    "round cannot be reused by another paper. An empty rule with no such sentence is read as an "
    "unfinished answer rather than as a finding, and the round teaches the library nothing. "
    "When the round ran a compile-time rule you are also given its implementation report: every "
    "object the rule was applied to with the value before and after, and every place the "
    "compiler could not reach. Read it before you explain the round, then answer attribution "
    "with one of four words -- rule_is_wrong when the way of drawing the rule states is itself "
    "why the figure got worse; not_carried_out when the rule was right but the compiler did not "
    "carry it out where it mattered; local_side_effect when the rule was carried out but broke "
    "something else; unmeasurable when the reading itself is not trustworthy. "
    "not_carried_out is only accepted against the report, and local_side_effect only when the "
    "figure changed and the comparison named a defect; a claim the report does not support is "
    "thrown away. "
    "A round that moved the figure instead of running a rule is given that move's own "
    "evidence in the same place: whether the figure changed at all, and which objects it "
    "changed. Answer attribution against it exactly as you would against a rule's report -- "
    "not_carried_out when the move never reached the figure, local_side_effect when it did "
    "and the comparison named what it broke. "
    "When the answer is not_carried_out or local_side_effect, say what to do about it. "
    "refined_rule is the same rule written more precisely -- narrower, or with one parameter "
    "moved a notch. follow_up is one bounded repair aimed at the object the comparison faulted, "
    "and unlike rule it may name this figure's own object ids. follow_up may only name a repair "
    "that edits the figure program or a compile-time rule: a repair that re-runs a planning "
    "stage is refused here, because asking again is only worth it while it is cheap. Asking "
    "is not the same as deciding: the loop spends at most three attempts on one lesson and "
    "measures each of them, so name the attempt you believe in rather than the one that is "
    "easiest to write. "
    "One more field, and it is the one that carries a lesson to a later round: prompt_patch. "
    "Write it when you can say which stage's prompt was wrong and what it should have said "
    "instead -- apply_at names that stage, directive is one sentence saying which requirement "
    "was not written and which way to change it, and condition says when it applies (the defect "
    "patterns it is about, and a metric with op and value when the round measured a bound). "
    "When the round's own review named the fault, it arrives as fault_attribution carrying that "
    "stage and the requirement that went unwritten, and the patch is that requirement said again "
    "for a later prompt: write it out as directive and set apply_at to that stage. A mechanism "
    "rule is not a substitute, because a rule says what to draw and the patch says which prompt "
    "should have asked for it -- so a round that named a fault has already written its patch. "
    "When prompt_patch_context arrives with this round it is that stage's own record of what is "
    "already asked for: the requirements that stage owns, the sentences already standing there, and "
    "the sentences the library already holds. Write the requirement that is missing from it. A "
    "sentence that is already in already_in_library is not a finding -- it is folded into the row "
    "that already carries it, so writing it again buys nothing. Write the different one, or leave "
    "the patch empty and say in why_changed that this stage has nothing left unwritten that you can "
    "name. "
    "directive must name no node label, no object id and no coordinate; it is the sentence "
    "another figure would be drawn by. Write prompt_patch only from what this round measured or "
    "saw: a patch is injected into a later prompt on the strength of the measurement behind this "
    "row, so a sentence you cannot stand behind costs an injection and buys nothing. Leave it as "
    "{} when this round taught nothing a later prompt should carry. "
    "Three fields filled in, as a patch you can write: "
    '{"apply_at": "ppt_compilation", "condition": {"defect_patterns": ["cross_panel"]}, '
    '"directive": "Route connectors on shared vertical channels so two arrows never cross between panels."}. '
    "And a rule you have just stated is one of these already: it says how a figure should be drawn, "
    "which is exactly a requirement some stage's prompt did not ask for. So a row that fills rule and "
    "leaves prompt_patch empty has written that sentence once and filed it nowhere -- fill in both. "
    "One more field, and only you can fill it: layer_settled. The figure is built one layer at a "
    "time -- the paper read, its content chosen, the structure planned, the theme chosen, the icons "
    "drawn, the composition built -- and this round has just looked at one of them. When a layer is "
    "finished with, so that planning it again would not make the figure better, say so: "
    '{"stage": "<the layer>", "because": "<the reading, or the thing you saw, that says so>"}. '
    "When the round carries a layer_plan, the layer it names as working_on is the one being asked "
    "about: answer for it, or for a layer below it that this round has finished with. A layer you "
    "name as good is one the next round leaves alone and works below, which is how the figure moves "
    "on -- naming a layer good is what carries the round down the pipeline, not a way of stopping. "
    "Leave it {} "
    "only when the layer being asked about is not good yet, and then say what is still wrong "
    "with it -- an empty answer is the statement that the work is not over, not a way of saying "
    "nothing about it, and a round that stays silent every time leaves the next round to work on "
    "whatever looks worst. The loop does not decide this for you and nothing in it ever will: the "
    "numbers you are shown describe the figure, they do not say a layer is done with, and settling "
    "one moves the next round's work one layer down. "
    "A person's own instruction, when there is one, arrives as human_preferences: each names a stage, "
    "the condition it applies under and the sentence that stage's prompt should have carried. It "
    "outranks your own taste about how the figure looks -- follow it, and when this round's reading "
    "says it is the wrong instruction, say so in why_changed with the reading that says it rather "
    "than quietly ignoring it. "
    "Every substantive sentence you write about why this round went the way it did belongs "
    "in claims as well, so a later round can settle it instead of taking your word for it. "
    "Write each as {claim, apply_at, prediction, criterion}: claim is the sentence itself, "
    "apply_at names the stage it is about, and criterion is the reading that would settle it, "
    "given as {metric, op, value}. metric must name a number this loop measures -- a defect "
    "kind such as interface:box_pass, a dimension such as dimension:readability, or "
    "overall_score -- op is one of <= < >= > == !=, and value is the number you expect. A "
    "sentence with no criterion is kept as a note and settled by nobody, so give a criterion "
    "whenever the round measured something you can name. Do not restate the same claim twice."
)


ADVISORY_COMPARISON_RULE = (
    "Two images are supplied after this text, both the same figure from the same paper: the first is A, "
    "the version the loop has already accepted, and the second is B, this round's candidate, which the "
    "candidate block below describes. Answer the pair before you answer B alone: set preferred_candidate "
    "to A, B or tie, where B means this round's candidate is the better of the two. Then name, in diagnosis, "
    "what the one you did not prefer shows -- figure names which of the two you are faulting, type must come "
    "from the diagnosis menu, and evidence must be something visible in that image. A defect you cannot point "
    "at is not evidence, so do not invent one to fill a field and do not restate the same defect twice. Every "
    "stage_feedback entry must restate one of those diagnosis entries for B, so when you prefer B over A the "
    "list is empty. " + WHOLE_FIRST_RULE
)


def _advisory_comparison_schema() -> dict[str, Any]:
    """The advisory schema plus the two fields the comparison question asks for.

    Both fields are read out of :func:`judgement_schema`, which is the training contract's own
    vocabulary, so the menu a served question offers cannot drift from the menu a trained answer
    was labelled with.
    """

    pairwise = judgement_schema(PAIRWISE_TASK_TYPES[0])
    schema: dict[str, Any] = {"preferred_candidate": pairwise["preferred_candidate"]}
    schema.update(ADVISORY_OUTPUT_SCHEMA)
    schema["diagnosis"] = pairwise["diagnosis"]
    return schema


def output_schema(task_type: str, *, role: str = "", comparison: bool = False) -> dict[str, Any]:
    """Return the required output fields of one Judge task for one role.

    A repair judgement answers a role question, so the two roles get two schemas: the
    advisory schema asks for the defect and its bounded repair, the acceptance schema asks
    for the score and the blockers. ``role`` decides which one is asked for.

    ``comparison`` marks the one variant: an advisory repair asked about two figures must also
    say which of them it prefers and which defect class it faults, so that question carries its
    own schema. It is False everywhere else, and only the advisory family may set it -- a
    measurement role offered a preference stops being a measurement.
    """

    if task_type == PAIRWISE_TASK_TYPE:
        return PAIRWISE_ACCEPTANCE_OUTPUT_SCHEMA
    if task_type == "repair":
        family = judge_role_family(role)
        if comparison and family == ADVISORY_ROLE_FAMILY:
            return _advisory_comparison_schema()
        return REPAIR_OUTPUT_SCHEMAS[family]
    if task_type in PROCESS_TASK_TYPES:
        return PROCESS_OUTPUT_SCHEMAS[task_type]
    return judgement_schema(task_type)


def build_reflection_prompt(outcome: dict[str, Any]) -> str:
    """Render the reflection prompt from the same source as the Judge prompt."""

    return REFLECTION_PROMPT + "\nOUTCOME=" + json.dumps(outcome, ensure_ascii=False)


def judgement_target(judgement: dict[str, Any], *, role: str = "") -> dict[str, Any]:
    """Return exactly the fields the served schema asks for, for one role.

    A label built here cannot drift from the served schema, and it can never carry the
    loop's own bookkeeping (``repair``, ``contract``, stage/severity) or a consequence
    the Judge only learns after it has spoken.
    An acceptance label therefore carries a score and no stage feedback; an advisory label
    carries the feedback and no score.
    """

    source = judgement if isinstance(judgement, dict) else {}
    schema = output_schema("repair", role=role)
    target = {field: source.get(field) for field in schema if field != "stage_feedback"}
    if "stage_feedback" in schema:
        fields = schema["stage_feedback"][0]
        target["stage_feedback"] = [
            {field: item.get(field) for field in fields}
            for item in source.get("stage_feedback") or []
            if isinstance(item, dict)
        ]
    return target


def build_prompt(row: dict[str, Any]) -> str:
    """Render one Judge prompt.

    ``process_context`` and ``candidate`` carry only what is knowable when the Judge is
    called. Unseen consequences stay in the example's ``observed`` block so a prompt can
    never leak the answer it is asking for. ``ground_truth.reference_figure``, when the
    sample carries one, is handed over verbatim: what the author's own figure draws is a
    fact the Judge is allowed to see, and a row without it keeps the exact v006 payload.
    """

    task = str(row.get("task_type") or "")
    paper = row.get("paper") if isinstance(row.get("paper"), dict) else {}
    gt = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    evidence = paper.get("evidence_spans") or paper.get("evidence") or []
    if len(evidence) > 12:
        evidence = evidence[:12]
    # A dataset row records the role that produced its labels in ``judge_metadata``, an online
    # row carries it directly. Both name the same thing, so one reader resolves both.
    metadata = row.get("judge_metadata") if isinstance(row.get("judge_metadata"), dict) else {}
    role = str(row.get("judge_role") or metadata.get("judge_role") or "")
    family = judge_role_family(role, frozen=bool(row.get("frozen")))
    pairwise = task == PAIRWISE_TASK_TYPE
    identity = PAIRWISE_ACCEPTANCE_IDENTITY if pairwise else ACCEPTANCE_IDENTITY if family == ACCEPTANCE_ROLE_FAMILY else ADVISORY_IDENTITY
    catalog = ""
    if family == ADVISORY_ROLE_FAMILY and not pairwise:
        # The seat is asked where a defect entered, so it is handed the table it answers from in
        # the same call as the action catalog: the two menus are rendered from the contract the
        # gate later checks the answer against, so the served question and the judged one are one
        # text rather than two that drift.
        catalog = render_action_catalog_for_prompt() + "\n" + render_fault_attribution_for_prompt() + "\n"
    # A second image only ever travels with the advisory role. The acceptance and pairwise roles
    # measure one state, and a reference handed to a measurement is a preference in its mouth.
    context = row.get("process_context") if isinstance(row.get("process_context"), dict) else {}
    comparison = bool(context.get("comparison")) and family == ADVISORY_ROLE_FAMILY and not pairwise
    payload = {
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "domain": row.get("domain") or labels.get("domain"),
        "domain_rubric": row.get("domain_rubric") or labels.get("domain_rubric"),
        "method_summary": gt.get("method_summary"),
        "required_entities": gt.get("required_entities") or gt.get("entities"),
        "required_relations": gt.get("required_relations") or gt.get("relations"),
        "evidence_spans": evidence,
        "task_type": task,
        "judge_role": family,
        "candidate": row.get("candidate") if isinstance(row.get("candidate"), dict) else {},
        "process_context": row.get("process_context") if task in PROCESS_TASK_TYPES else {},
    }
    reference_figure = gt.get("reference_figure")
    if isinstance(reference_figure, dict) and reference_figure:
        payload["reference_figure"] = reference_figure
    pair = ""
    if pairwise:
        pair = "\nTwo images are supplied after this text, both of the same figure from the same paper: the first is A, the state the loop already accepted, and the second is B, the state it has just produced. " + PAIRWISE_MARGIN_RULE + " " + WHOLE_FIRST_RULE + "\n"
    elif comparison:
        pair = "\n" + ADVISORY_COMPARISON_RULE + "\n"
    elif task in PAIRWISE_TASK_TYPES:
        pair = "\nTwo images are supplied after this text: the first is candidate A and the second is candidate B.\n"
    # The promoted patches join below the reference contract and above the output schema,
    # so they refine the instructions without ever restating the schema they must obey.
    patches = row.get("prompt_patches")
    if not patches and isinstance(payload.get("process_context"), dict):
        patches = payload["process_context"].get("prompt_patches")
    head = (
        f"{identity}\n"
        f"{catalog}"
        "Return only one valid JSON object, with no markdown.\n"
        f"Judge role: {family}.\n"
        f"Task: {task}.\n{pair}Reference contract:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n"
    )
    return head + render_patch_block(patches) + f"Required output schema (types are descriptive):\n{json.dumps(output_schema(task, role=role, comparison=comparison), ensure_ascii=False)}"
