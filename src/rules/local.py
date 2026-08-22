"""Level 3 placeholder: blackbox / biologically-plausible local rules.

Δw_ij = R_φ(a_pre, a_post, w_ij, eligibility trace e_ij, modulator m_t, ...)

Deliberately empty for now (see PDF §1 and ROADMAP "Phase A"). When these
arrive they implement the same LearningRule interface — the kernel/loss/trainer
stack does not change. The main new plumbing they need is access to pre/post
activations per connection, which ProbedModel hooks can provide.
"""
