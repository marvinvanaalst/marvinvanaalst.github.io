---
title: 'Is Math Now a Search Problem?'
description: 'The same models that misjudge a trip to the car wash help solve hard open problems in mathematics. Maybe that stops being a paradox once you look at proving theorems as search.'
categories:
  - musings
date: '2026-11-21'
author: 'Marvin van Aalst'
layout: tutorials
published: true
---

# Is Math Now a Search Problem?

Over the last months we've watched something remarkable happen: mathematicians, working together with LLM agents, have cracked problems that had been sitting on the "hard" pile for a very long time. The human is still clearly in the loop, steering, but the agents are doing a lot more than proofreading.
And yet I keep running into a conundrum: the very same technology will cheerfully advise me to walk to the car wash around the corner, because it's not worth the drive. How can something that trips over that be super-humanly good at reasoning through hard math?

As usual with conundrums, it can help to take a step back and ask whether we're asking the right question to begin with. Maybe the problem is the word "reasoning". So here is a thought I want to play with: maybe a large part of modern math is best viewed as a search problem.

LLMs are, among other things, extremely capable search engines, and they've effectively read millennia of attempts at solving math problems. Every trick, substitution, clever lemma and failed approach someone bothered to write down. Of course, finding a proof by exhaustively combining all those approaches is hopeless, since the space explodes far too quickly. But LLMs don't search blindly. They have a sense of which approaches tend to go together, which turns an impossible exhaustive search into a biased one. That bias is far from perfect, but it doesn't need to be.

On top of that comes the mathematician. Their guidance does what a good advisor does for a PhD student: rule out whole regions of the search space ("that won't work, the bound is too weak") and point at the promising ones. That cuts the space down dramatically.

The last piece of the puzzle is feedback. These agents are trained with reinforcement learning, so they're used to trying things and being corrected. Proof assistants like Lean offer exactly that kind of correction while searching: every step either checks or it doesn't, so a wrong turn gets caught immediately instead of three pages later.

So the open question I'm left with is this: with all the context we've gathered over the last millennia, how much of math nowadays boils down to a (very hard) search problem?
