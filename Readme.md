**ALG_3PlanningAlgorithm — Hybrid RL-TLBO Scheduling Algorithm**

- **Location**: `sources/org/workflowsim/ALG_3PlanningAlgorithm.java`
- **Purpose**: Implements a hybrid scheduling/planning algorithm that combines a simple Reinforcement Learning (RL) agent for initial task-to-VM assignment with a Teaching–Learning-Based Optimization (TLBO) refinement phase to reduce makespan and balance VM loads.

**Overview**
- **What it does**: The algorithm assigns tasks (cloudlets) to virtual machines (VMs) in two stages:
  - Stage 1 — RL-based online assignment: uses an epsilon-greedy RL policy with a Q-table to choose a VM for each ready task and updates Q-values using inverse finish-time reward.
  - Stage 2 — TLBO optimization: performs repeated Teacher and Learner phases to refine the full allocation and reduce the simulated makespan.

**Key Components (in `ALG_3PlanningAlgorithm.java`)**
- `RLAgent`: maintains a Q-table, selects VMs via epsilon-greedy, updates Q-values, and decays epsilon.
- `TLBOOptimizer`: runs teacher and learner phases to refine an allocation; includes `simulateMakespan` and `simulateLocalLoad` helper methods.
- `VMInfo`: simple helper class to track predicted VM load (used during RL assignment).

**Tunable Parameters**
- `MAX_TLBO_ITER` (int): number of TLBO iterations (default 100). Lower for faster runs, higher to attempt better refinement.
- RLAgent fields: `learningRate`, `discountFactor`, `epsilon`, `epsilonDecay`. Tweak these to control learning speed and exploration.

**Behavior & Output**
- The algorithm prints progress messages to stdout (e.g., "Running Hybrid RL-TLBO Scheduling Algorithm").
- After optimization it prints per-VM loads and two summary numbers: an aggregated makespan and the maximum VM load.

**Integration / Usage**
- Build: compile this source as part of the WorkflowSim project (it extends `BasePlanningAlgorithm`). Ensure the class is included on the classpath when running your simulation.

- Minimal example (conceptual):
  ```java
  List<Vm> vmList = ...;       // prepare VM list
  List<Task> taskList = ...;   // prepare Task/Cloudlet list

  ALG_3PlanningAlgorithm alg = new ALG_3PlanningAlgorithm();
  alg.setVmList(vmList);       // inherited from BasePlanningAlgorithm
  alg.setTaskList(taskList);   // inherited from BasePlanningAlgorithm
  alg.run();                   // runs RL then TLBO, prints summary
  ```

- Embedding in WorkflowSim: If WorkflowSim uses a planner/strategy registry or configuration file, point the planner to use `ALG_3PlanningAlgorithm` (replace the planner class or call the algorithm directly from your scenario setup). If unsure, search for other `*PlanningAlgorithm` usages in the codebase to mirror how they are instantiated/used.

**Tuning Tips**
- If runs are slow, reduce `MAX_TLBO_ITER` and decrease the search-space used in teacher selection.
- To encourage exploration early, increase `epsilon` (e.g., 0.5) and tune `epsilonDecay` to control how quickly it anneals.
- For reproducible experiments, set a fixed `Random` seed in `RLAgent` and `TLBOOptimizer` constructors.

**Limitations & Notes**
- State representation used by `RLAgent` is compact/simple (task length + sorted VM loads). This keeps the Q-table small but may limit expressiveness.
- `TLBOOptimizer.findTeacherVM` tests full assignment alternatives which can be expensive if you have many VMs; consider approximating the teacher selection for large environments.
- The Q-table is in-memory only (no persistence across runs).

**References**
- The source file lists a few references used during implementation; see the top of `ALG_3PlanningAlgorithm.java` for the reference names (e.g., `Mathematics-11-03364.pdf`, `Hybrid_Teaching-Learning-Based_Optimization_for_Wo.pdf`, `2408.02938v1.pdf`).

**Next steps / Suggestions**
- Add unit tests that exercise the RL selection and TLBO phases on small synthetic workflows to validate improvements and to tune hyperparameters.
- Consider adding a configuration file or `Parameters` hook so the main simulation can tune `MAX_TLBO_ITER` and RL hyperparameters without code edits.

**Author / Location**
- Implemented in `ALG_3PlanningAlgorithm.java` (see path above). This README documents the algorithm's purpose, usage, and tuning notes.
