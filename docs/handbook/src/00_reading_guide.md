# How to read this handbook

This handbook explains **The Next Ravi Varma** from the ground up: what problem it solves, how every part works,
why it was built this way, how to run and debug it, and how to defend it in a review. Everything below is grounded in
the repository at commit `e2cae62` (22 October 2026). Every file path and symbol name refers to that tree.

## Evidence labels

The handbook separates what the code *proves* from what is *inferred*. You will see these labels:

| Label | Meaning |
|---|---|
| **Confirmed** | Stated in repository documentation or commit messages *and* visible in the implementation |
| **Implemented** | Directly visible in the code or configuration, even if no document explains why |
| **Measured** | A number produced by a script in this repository, with the output file named |
| **Inferred** | A reasonable interpretation of the code. The original motivation is not recorded anywhere |
| **Not implemented** | Mentioned (config, docstring, proposal) but absent or explicitly disabled in code |

## Chapter map

| Chapter | Question it answers |
|---|---|
| 1. Project overview | What is this, for whom, and what works? |
| 2. Technology stack | What does each library do here, and what are the alternatives? |
| 3. Architecture | How do the parts fit together? (diagrams) |
| 4. Features | How does each feature work, end to end? |
| 5. Decision register | Why was each important choice made, and when should it change? |
| 6. Codebase guide | What does each module do and who calls it? |
| 7. Data, interfaces and flows | What files, schemas, CLIs, app events and external APIs exist? |
| 8. First principles | The theory behind diffusion, LoRA, CLIP, ControlNet and the metrics |
| 9. Setup and operations | How to install, run, verify, debug and reset |
| 10. Testing, reliability, security | What is proven, what is risky, and what to fix |
| 11. Alternatives and improvements | What could be simpler, and what to change first |
| 12. Demonstration and interview prep | Scripts, 35 questions with answers, honest weaknesses |
| 13–14 | Glossary and a one-page quick reference |
| Appendices | Open questions and assumptions; the verification report |

## Scope of this document

This is **documentation and analysis only**: no application code, configuration or dependency was changed to write it.
Where the documentation in the repository disagrees with the code, the handbook says so and trusts the code. Those
cases are collected in Appendix A.
