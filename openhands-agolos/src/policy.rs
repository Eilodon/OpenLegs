//! Policy/EFE Calculator - Active Inference action selection
//!
//! Ported from Pandora's `policy.rs`
//! Provides Expected Free Energy based policy selection.

use std::collections::HashMap;

/// Action policy representing possible agent actions
#[derive(Debug, Clone, PartialEq)]
pub enum ActionPolicy {
    /// Execute the proposed action
    Execute,
    /// Request user confirmation before execution
    RequestConfirmation,
    /// Skip this action
    Skip,
    /// Fallback to safer alternative
    Fallback,
    /// Gather more information first
    Explore,
}

impl ActionPolicy {
    /// Check if this policy requires user interaction
    pub fn requires_interaction(&self) -> bool {
        matches!(self, ActionPolicy::RequestConfirmation)
    }

    /// Estimate intrusiveness level (0-1)
    pub fn intrusiveness(&self) -> f32 {
        match self {
            ActionPolicy::Execute => 0.3,
            ActionPolicy::RequestConfirmation => 0.6,
            ActionPolicy::Skip => 0.0,
            ActionPolicy::Fallback => 0.2,
            ActionPolicy::Explore => 0.1,
        }
    }
}

/// Policy evaluation with EFE components
#[derive(Debug, Clone)]
pub struct PolicyEvaluation {
    pub policy: ActionPolicy,
    /// Pragmatic value (goal alignment, higher = better)
    pub pragmatic_value: f32,
    /// Epistemic value (information gain, higher = more learning)
    pub epistemic_value: f32,
    /// Expected Free Energy (lower = better policy)
    pub efe: f32,
    /// Selection probability after softmax
    pub probability: f32,
}

impl PolicyEvaluation {
    pub fn new(policy: ActionPolicy, pragmatic: f32, epistemic: f32) -> Self {
        // EFE = -pragmatic - epistemic (we want to maximize both)
        let efe = -pragmatic - epistemic;
        Self {
            policy,
            pragmatic_value: pragmatic,
            epistemic_value: epistemic,
            efe,
            probability: 0.0,
        }
    }
}

/// Expected Free Energy Calculator
#[derive(Debug, Clone)]
pub struct EFECalculator {
    /// Temperature for softmax (higher = more exploration)
    pub temperature: f32,
    /// Weight for epistemic (exploration) component
    pub epistemic_weight: f32,
    /// Weight for pragmatic (exploitation) component
    pub pragmatic_weight: f32,
}

impl Default for EFECalculator {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            epistemic_weight: 0.3,
            pragmatic_weight: 0.7,
        }
    }
}

impl EFECalculator {
    pub fn new(temperature: f32, epistemic_weight: f32) -> Self {
        Self {
            temperature,
            epistemic_weight,
            pragmatic_weight: 1.0 - epistemic_weight,
        }
    }

    /// Compute weighted EFE
    pub fn compute_efe(&self, pragmatic: f32, epistemic: f32) -> f32 {
        -(self.pragmatic_weight * pragmatic + self.epistemic_weight * epistemic)
    }

    /// Select policy using softmax over negative EFE values
    pub fn select_policy<'a>(&self, evaluations: &'a mut [PolicyEvaluation]) -> Option<&'a PolicyEvaluation> {
        if evaluations.is_empty() {
            return None;
        }

        // Compute softmax probabilities
        let max_neg_efe = evaluations
            .iter()
            .map(|e| -e.efe)
            .fold(f32::NEG_INFINITY, f32::max);

        let exp_sum: f32 = evaluations
            .iter()
            .map(|e| ((-e.efe - max_neg_efe) / self.temperature).exp())
            .sum();

        for eval in evaluations.iter_mut() {
            eval.probability = ((-eval.efe - max_neg_efe) / self.temperature).exp() / exp_sum;
        }

        // Sort by probability (highest first)
        evaluations.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        evaluations.first()
    }

    /// Evaluate action in context
    pub fn evaluate_action(
        &self,
        action_type: &str,
        goal_alignment: f32,
        uncertainty: f32,
        past_success_rate: f32,
    ) -> PolicyEvaluation {
        // Pragmatic: how well does this align with goal?
        let pragmatic = goal_alignment * past_success_rate;

        // Epistemic: how much can we learn? (higher uncertainty = more to learn)
        let epistemic = uncertainty * 0.5;

        let policy = if goal_alignment < 0.2 {
            ActionPolicy::Skip
        } else if uncertainty > 0.8 {
            ActionPolicy::Explore
        } else if past_success_rate < 0.3 {
            ActionPolicy::Fallback
        } else if goal_alignment > 0.8 && past_success_rate > 0.7 {
            ActionPolicy::Execute
        } else {
            ActionPolicy::RequestConfirmation
        };

        let mut eval = PolicyEvaluation::new(policy, pragmatic, epistemic);
        eval.efe = self.compute_efe(pragmatic, epistemic);
        eval
    }
}

/// Policy selector for agent actions
pub struct PolicySelector {
    calculator: EFECalculator,
    action_history: HashMap<String, ActionStats>,
}

#[derive(Debug, Clone, Default)]
struct ActionStats {
    success_count: u32,
    failure_count: u32,
    total_count: u32,
}

impl ActionStats {
    fn success_rate(&self) -> f32 {
        if self.total_count == 0 {
            0.5 // Prior
        } else {
            self.success_count as f32 / self.total_count as f32
        }
    }
}

impl Default for PolicySelector {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicySelector {
    pub fn new() -> Self {
        Self {
            calculator: EFECalculator::default(),
            action_history: HashMap::new(),
        }
    }

    pub fn with_calculator(calculator: EFECalculator) -> Self {
        Self {
            calculator,
            action_history: HashMap::new(),
        }
    }

    /// Select best policy for an action
    pub fn select(
        &self,
        action_type: &str,
        goal_alignment: f32,
        uncertainty: f32,
    ) -> PolicyEvaluation {
        let stats = self.action_history.get(action_type).cloned().unwrap_or_default();
        self.calculator.evaluate_action(
            action_type,
            goal_alignment,
            uncertainty,
            stats.success_rate(),
        )
    }

    /// Record action outcome for learning
    pub fn record_outcome(&mut self, action_type: &str, success: bool) {
        let stats = self.action_history.entry(action_type.to_string()).or_default();
        stats.total_count += 1;
        if success {
            stats.success_count += 1;
        } else {
            stats.failure_count += 1;
        }
    }

    /// Get success rate for action type
    pub fn get_success_rate(&self, action_type: &str) -> f32 {
        self.action_history
            .get(action_type)
            .map(|s| s.success_rate())
            .unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_efe_calculation() {
        let calc = EFECalculator::default();

        // Higher pragmatic = lower EFE (better)
        let efe1 = calc.compute_efe(0.8, 0.2);
        let efe2 = calc.compute_efe(0.3, 0.2);
        assert!(efe1 < efe2);
    }

    #[test]
    fn test_policy_selection() {
        let calc = EFECalculator::default();

        let mut evaluations = vec![
            PolicyEvaluation::new(ActionPolicy::Execute, 0.9, 0.1),
            PolicyEvaluation::new(ActionPolicy::Skip, 0.1, 0.1),
            PolicyEvaluation::new(ActionPolicy::Explore, 0.5, 0.8),
        ];

        let selected = calc.select_policy(&mut evaluations).unwrap();

        // Best policy should have highest probability
        assert!(selected.probability > 0.3);
        // Probabilities should sum to ~1
        let total_prob: f32 = evaluations.iter().map(|e| e.probability).sum();
        assert!((total_prob - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_selector_learning() {
        let mut selector = PolicySelector::new();

        // Initially unknown
        assert_eq!(selector.get_success_rate("test_action"), 0.5);

        // Record outcomes
        selector.record_outcome("test_action", true);
        selector.record_outcome("test_action", true);
        selector.record_outcome("test_action", false);

        // Should be 2/3 success rate
        let rate = selector.get_success_rate("test_action");
        assert!((rate - 0.666).abs() < 0.01);
    }
}
