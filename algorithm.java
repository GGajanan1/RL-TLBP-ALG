  package org.workflowsim;

import java.util.*;
import org.cloudbus.cloudsim.Vm;
import org.workflowsim.planning.BasePlanningAlgorithm;
import org.workflowsim.utils.Parameters;

/**
 * Hybrid RL-TLBO Scheduling Algorithm.
 *
 * This implementation integrates:
 * 1. Reinforcement Learning (RL) for initial task-to-VM assignment using adaptive exploration.
 * 2. A modified Teaching–Learning-Based Optimization (TLBO) that refines the allocation through
 *    a teacher phase (global improvement) and a learner phase (local pairwise improvement).
 *
 * References:
 * - Mathematics-11-03364.pdf
 * - Hybrid_Teaching-Learning-Based_Optimization_for_Wo.pdf
 * - 2408.02938v1.pdf
 */
public class ALG_3PlanningAlgorithm extends BasePlanningAlgorithm {
    // Store task timeline data: task id -> {start time, finish time}
    private Map<Integer, double[]> taskTimeline = new HashMap<>();

    // Maximum number of global optimization iterations (TLBO iterations)
    private static final int MAX_TLBO_ITER = 100;

    @Override
    public void run() {
        System.out.println("Running Hybrid RL-TLBO Scheduling Algorithm");

        List<Vm> vmList = getVmList();
        // Create a modifiable task list from the original task list
        List<Task> remainingTasks = new ArrayList<>(getTaskList());
        int taskNum = remainingTasks.size();

        // Allocation: allocation[i] holds the VM id for the task at index i
        // We maintain a consistent ordering by recording tasks in a separate list.
        List<Task> allTasks = new ArrayList<>(remainingTasks);
        int[] allocation = new int[allTasks.size()];

        // Priority queue to track each VM’s current finish time (load)
        PriorityQueue<VMInfo> vmQueue = new PriorityQueue<>(Comparator.comparingDouble(v -> v.currentLoad));
        for (int i = 0; i < vmList.size(); i++) {
            vmQueue.add(new VMInfo(i, 0.0));
        }

        // Create RL agent and TLBO optimizer
        RLAgent rlAgent = new RLAgent(vmList);
        TLBOOptimizer tlboOptimizer = new TLBOOptimizer(vmList, allTasks);

        // ***********************
        // Stage 1: RL-based Scheduling with dynamic task list update
        // ***********************
        List<Task> readyList = new ArrayList<>();
        while (!remainingTasks.isEmpty()) {
            readyList.clear();

            // Identify ready tasks (tasks with no pending dependencies).
            for (Task task : remainingTasks) {
                boolean ready = true;
                for (Task parent : task.getParentList()) {
                    // If a task's parent is still unscheduled (present in remainingTasks), then it's not ready.
                    if (remainingTasks.contains(parent)) {
                        ready = false;
                        break;
                    }
                }
                if (ready) {
                    readyList.add(task);
                }
            }
        

            // Process each ready task.
            for (Task task : readyList) {
                // Get the index of the task in the overall task list.
                int taskIndex = allTasks.indexOf(task);
                int selectedVmIndex = rlAgent.selectVM(task, vmQueue);
                allocation[taskIndex] = selectedVmIndex;
                task.setVmId(selectedVmIndex);

                // Compute runtime on selected VM.
                Vm vm = vmList.get(selectedVmIndex);
                double runtime = task.getCloudletLength() / vm.getMips();
                VMInfo vmInfo = rlAgent.getVMInfo(vmQueue, selectedVmIndex);
                double startTime = vmInfo.currentLoad;
                double finishTime = startTime + runtime;

                // Update VM load and record timeline.
                vmQueue.remove(vmInfo);
                vmInfo.currentLoad = finishTime;
                vmQueue.add(vmInfo);
                taskTimeline.put(task.getCloudletId(), new double[]{startTime, finishTime});

                // Update the Q-table based on the finish time reward.
                rlAgent.updateQValue(task, selectedVmIndex, finishTime);
                // Decay exploration rate after each decision.
                rlAgent.decayEpsilon();
            }
            // Remove ready tasks from the remainingTasks list.
            remainingTasks.removeAll(readyList);
        }

        // ***********************
        // Stage 2: Global TLBO-based Optimization
        // ***********************
        // Run TLBO optimization for a fixed number of iterations.
        for (int iter = 0; iter < MAX_TLBO_ITER; iter++) {
            // Teacher Phase: Move task assignments toward the global best (teacher).
            allocation = tlboOptimizer.teacherPhase(allocation);
            // Learner Phase: Refine allocation through pairwise comparisons.
            allocation = tlboOptimizer.learnerPhase(allocation);
        }

        // Display the overall makespan after optimization.
        displayMakespan(allocation, vmList, allTasks);
    }

    /**
     * Display the overall makespan by simulating the VM loads using the final allocation.
     * Here we iterate over tasks using the consistent ordering of the allocation array.
     */
    private void displayMakespan(int[] allocation, List<Vm> vmList, List<Task> taskList) {
        double[] loads = new double[vmList.size()];
        // For each task (using its index in the task list), accumulate the runtime on its assigned VM.
        for (int i = 0; i < taskList.size(); i++) {
            Task task = taskList.get(i);
            int vmIndex = allocation[i];
            Vm vm = vmList.get(vmIndex);
            loads[vmIndex] += task.getCloudletLength() / vm.getMips();
            
        }
        double maxLoad = 0.0,makeSpan=0.0;
        for (double load : loads) {
        	System.out.print(load+" ");
            makeSpan+=load;
            if(load>maxLoad) {
            	maxLoad=load;
            }
        }
        System.out.println("Overall Makespan Time: " + makeSpan);
        System.out.println("MaxLoade on VM is: " + maxLoad);
    }

    /**
     * Helper class to track each VM's current workload (predicted finish time).
     */
    class VMInfo {
        int vmId;
        double currentLoad;

        public VMInfo(int vmId, double currentLoad) {
            this.vmId = vmId;
            this.currentLoad = currentLoad;
        }
    }

    /**
     * Reinforcement Learning Agent that selects a VM for a given task.
     * Uses a Q-table based on the state (combination of task length and sorted VM loads).
     */
    class RLAgent {
        private List<Vm> vmList;
        private Map<String, Double> qTable;
        private Random random;
        private double learningRate = 0.1;
        private double discountFactor = 0.9;
        private double epsilon = 0.3;       // Initial exploration rate
        private double epsilonDecay = 0.95; // Decay rate per update

        public RLAgent(List<Vm> vmList) {
            this.vmList = vmList;
            this.qTable = new HashMap<>();
            this.random = new Random();
        }

        /**
         * Select a VM based on the current state using an epsilon-greedy policy.
         */
        public int selectVM(Task task, PriorityQueue<VMInfo> vmQueue) {
            String state = getState(task, vmQueue);
            int bestVmIndex = 0;
            if (random.nextDouble() < epsilon) {
                // Exploration: choose a random VM.
                bestVmIndex = random.nextInt(vmList.size());
            } else {
                // Exploitation: select the VM with the highest Q-value.
                double maxQ = Double.NEGATIVE_INFINITY;
                for (int i = 0; i < vmList.size(); i++) {
                    double q = qTable.getOrDefault(state + "_" + i, 0.0);
                    if (q > maxQ) {
                        maxQ = q;
                        bestVmIndex = i;
                    }
                }
            }
            return bestVmIndex;
        }

        /**
         * Update the Q-value based on the finish time reward.
         */
        public void updateQValue(Task task, int vmIndex, double finishTime) {
            String state = getState(task, new ArrayList<>(vmList), finishTime);
            double reward = 1.0 / finishTime; // Inverse of finish time as reward
            double currentQ = qTable.getOrDefault(state + "_" + vmIndex, 0.0);
            double maxFutureQ = getMaxQ(state);
            double newQ = currentQ + learningRate * (reward + discountFactor * maxFutureQ - currentQ);
            qTable.put(state + "_" + vmIndex, newQ);
        }

        /**
         * Get the maximum Q-value for the given state.
         */
        private double getMaxQ(String state) {
            double maxQ = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < vmList.size(); i++) {
                double q = qTable.getOrDefault(state + "_" + i, 0.0);
                if (q > maxQ) {
                    maxQ = q;
                }
            }
            return maxQ;
        }

        /**
         * Build a state string representation using task length and sorted VM loads.
         */
        private String getState(Task task, PriorityQueue<VMInfo> vmQueue) {
            StringBuilder sb = new StringBuilder();
            sb.append(task.getCloudletLength()).append("_");
            List<VMInfo> sortedList = new ArrayList<>(vmQueue);
            sortedList.sort(Comparator.comparingDouble(v -> v.currentLoad));
            for (VMInfo info : sortedList) {
                sb.append(info.currentLoad).append("_");
            }
            return sb.toString();
        }

        /**
         * Alternative state representation including the finish time.
         */
        private String getState(Task task, List<Vm> vmList, double finishTime) {
            return task.getCloudletLength() + "_" + finishTime;
        }

        /**
         * Retrieve the VMInfo for the given VM id from the queue.
         */
        public VMInfo getVMInfo(PriorityQueue<VMInfo> vmQueue, int vmId) {
            for (VMInfo info : vmQueue) {
                if (info.vmId == vmId) {
                    return info;
                }
            }
            return null;
        }

        /**
         * Decay the epsilon value to reduce exploration over time.
         */
        public void decayEpsilon() {
            epsilon = Math.max(0.01, epsilon * epsilonDecay);
        }
    }

    /**
     * Modified TLBO Optimizer that refines the allocation.
     * It performs two phases:
     * 1. Teacher Phase: Moves task assignments toward the global best (teacher vector).
     * 2. Learner Phase: Uses pairwise comparisons between tasks for local improvements.
     */
    class TLBOOptimizer {
        private List<Vm> vmList;
        private List<Task> taskList;
        private Random random;

        public TLBOOptimizer(List<Vm> vmList, List<Task> taskList) {
            this.vmList = vmList;
            this.taskList = taskList;
            this.random = new Random();
        }

        /**
         * Teacher Phase: For each task, try to adjust the assignment toward the teacher (global best).
         * The teacher is defined as the VM that, if assigned to all tasks, minimizes the simulated makespan.
         */
        public int[] teacherPhase(int[] allocation) {
            int[] newAllocation = allocation.clone();
            // Identify the teacher VM by testing full assignment alternatives.
            int teacherVm = findTeacherVM(newAllocation);
            for (int i = 0; i < taskList.size(); i++) {
                // For each task, if reassigning it to the teacher VM improves the makespan, update the allocation.
                int originalVm = newAllocation[i];
                newAllocation[i] = teacherVm;
                double newMakespan = simulateMakespan(newAllocation);
                double originalMakespan = simulateMakespan(allocation);
                if (newMakespan >= originalMakespan) {
                    // Revert if no improvement.
                    newAllocation[i] = originalVm;
                }
            }
            return newAllocation;
        }

        /**
         * Learner Phase: For each task, randomly pick a partner task and update the assignment
         * if the partner's assignment yields a lower local load.
         */
        public int[] learnerPhase(int[] allocation) {
            int[] newAllocation = allocation.clone();
            for (int i = 0; i < taskList.size(); i++) {
                // Select a random partner task (different from i)
                int j = random.nextInt(taskList.size());
                while (j == i) {
                    j = random.nextInt(taskList.size());
                }
                int vmI = newAllocation[i];
                int vmJ = newAllocation[j];
                // Evaluate local loads if task i were assigned to vmJ
                newAllocation[i] = vmJ;
                double newLoad = simulateLocalLoad(newAllocation, i);
                newAllocation[i] = vmI;
                double currentLoad = simulateLocalLoad(newAllocation, i);
                // If the partner’s assignment improves the local load, update the allocation.
                if (newLoad < currentLoad) {
                    newAllocation[i] = vmJ;
                }
            }
            return newAllocation;
        }

        /**
         * Find the teacher VM, defined as the VM that minimizes the simulated makespan if
         * all tasks were assigned to it.
         */
        	private int findTeacherVM(int[] allocation) {
	            int bestVm = 0;
	            double bestMakespan = Double.MAX_VALUE;
	            for (int vmIndex = 0; vmIndex < vmList.size(); vmIndex++) {
	                int[] testAllocation = new int[allocation.length];
	                Arrays.fill(testAllocation, vmIndex);
	                double makespan = simulateMakespan(testAllocation);
	                if (makespan < bestMakespan) {
	                    bestMakespan = makespan;
	                    bestVm = vmIndex;
	                }
	            }
	            return bestVm;
        	}

        /**
         * Simulate the overall makespan (maximum finish time) for the given allocation.
         */
        private double simulateMakespan(int[] allocation) {
            double[] loads = new double[vmList.size()];
            for (int i = 0; i < taskList.size(); i++) {
                Task task = taskList.get(i);
                int vmIndex = allocation[i];
                Vm vm = vmList.get(vmIndex);
                loads[vmIndex] += task.getCloudletLength() / vm.getMips();
            }
            double maxLoad = 0.0;
            for (double load : loads) {
                if (load > maxLoad) {
                    maxLoad = load;
                }
            }
            return maxLoad;
        }

        /**
         * Simulate the local load for a specific task given the current allocation.
         * This function evaluates the impact of the task's assignment on the load of its assigned VM.
         */
        private double simulateLocalLoad(int[] allocation, int taskIndex) {
            int vmIndex = allocation[taskIndex];
            double load = 0.0;
            // Only consider tasks assigned to the same VM.
            for (int i = 0; i < taskList.size(); i++) {
                if (allocation[i] == vmIndex) {
                    Task task = taskList.get(i);
                    Vm vm = vmList.get(vmIndex);
                    load += task.getCloudletLength() / vm.getMips();
                }
            }
            return load;
        }
    }
}
