//! Decision Tree for context-aware routing
//!
//! Ported from Pandora's `decision_tree.rs`.
//! Routes actions based on resource constraints and context.

/// Decision context for tree evaluation
#[derive(Debug, Clone, Default)]
pub struct DecisionContext {
    /// Remaining API budget (0.0-1.0)
    pub budget_remaining: f32,
    /// Context window usage (0.0-1.0)
    pub context_usage: f32,
    /// Current failure count
    pub failure_count: u32,
    /// Whether rate limited
    pub is_rate_limited: bool,
    /// Confidence in current action
    pub confidence: f32,
    /// Time since last action (ms)
    pub time_since_last_ms: u64,
    /// Whether this is a high priority task
    pub is_high_priority: bool,
}

impl DecisionContext {
    pub fn new(budget: f32, context: f32, confidence: f32) -> Self {
        Self {
            budget_remaining: budget,
            context_usage: context,
            confidence,
            ..Default::default()
        }
    }

    /// Is budget critically low (<10%)?
    pub fn is_budget_critical(&self) -> bool {
        self.budget_remaining < 0.1
    }

    /// Is context window nearly full (>90%)?
    pub fn is_context_constrained(&self) -> bool {
        self.context_usage > 0.9
    }

    /// Should we use resource-saving mode?
    pub fn should_save_resources(&self) -> bool {
        self.is_budget_critical() || self.is_context_constrained()
    }
}

/// Conditions for decision nodes
#[derive(Debug, Clone)]
pub enum Condition {
    /// Budget below threshold
    BudgetBelow(f32),
    /// Context usage above threshold
    ContextAbove(f32),
    /// Rate limited
    RateLimited,
    /// Failure count above threshold
    FailuresAbove(u32),
    /// Confidence below threshold
    ConfidenceBelow(f32),
    /// Time since last action above threshold (ms)
    IdleAbove(u64),
    /// High priority task
    HighPriority,
    /// Always true
    Always,
}

impl Condition {
    pub fn evaluate(&self, ctx: &DecisionContext) -> bool {
        match self {
            Condition::BudgetBelow(t) => ctx.budget_remaining < *t,
            Condition::ContextAbove(t) => ctx.context_usage > *t,
            Condition::RateLimited => ctx.is_rate_limited,
            Condition::FailuresAbove(t) => ctx.failure_count > *t,
            Condition::ConfidenceBelow(t) => ctx.confidence < *t,
            Condition::IdleAbove(t) => ctx.time_since_last_ms > *t,
            Condition::HighPriority => ctx.is_high_priority,
            Condition::Always => true,
        }
    }
}

/// Actions that can be taken
#[derive(Debug, Clone, PartialEq)]
pub enum DecisionAction {
    /// Full computation with all resources
    FullComputation,
    /// Simplified/cheaper computation
    SimplifiedComputation,
    /// Use cached result if available
    UseCached,
    /// Skip this action entirely
    Skip,
    /// Fallback to safe default
    SafeFallback,
    /// Request human intervention
    RequestHelp,
    /// Wait and retry later
    WaitRetry,
}

/// A node in the decision tree
#[derive(Debug, Clone)]
pub enum DecisionNode {
    /// Leaf node with action
    Leaf(DecisionAction),
    /// Branch node with condition
    Branch {
        condition: Condition,
        if_true: Box<DecisionNode>,
        if_false: Box<DecisionNode>,
    },
}

impl DecisionNode {
    pub fn leaf(action: DecisionAction) -> Self {
        DecisionNode::Leaf(action)
    }

    pub fn branch(condition: Condition, if_true: DecisionNode, if_false: DecisionNode) -> Self {
        DecisionNode::Branch {
            condition,
            if_true: Box::new(if_true),
            if_false: Box::new(if_false),
        }
    }
}

/// Decision tree for routing
pub struct DecisionTree {
    root: DecisionNode,
}

impl DecisionTree {
    pub fn new(root: DecisionNode) -> Self {
        Self { root }
    }

    /// Default tree for OpenHands agent
    ///
    /// Tree structure:
    /// [FailureCount > 3?]
    ///     YES → SafeFallback
    ///     NO → [Budget < 10%?]
    ///         YES → SimplifiedComputation
    ///         NO → [RateLimited?]
    ///             YES → WaitRetry
    ///             NO → [Context > 90%?]
    ///                 YES → UseCached
    ///                 NO → [Confidence < 0.3?]
    ///                     YES → RequestHelp
    ///                     NO → FullComputation
    pub fn default_for_agent() -> Self {
        let root = DecisionNode::branch(
            Condition::FailuresAbove(3),
            DecisionNode::leaf(DecisionAction::SafeFallback),
            DecisionNode::branch(
                Condition::BudgetBelow(0.1),
                DecisionNode::leaf(DecisionAction::SimplifiedComputation),
                DecisionNode::branch(
                    Condition::RateLimited,
                    DecisionNode::leaf(DecisionAction::WaitRetry),
                    DecisionNode::branch(
                        Condition::ContextAbove(0.9),
                        DecisionNode::leaf(DecisionAction::UseCached),
                        DecisionNode::branch(
                            Condition::ConfidenceBelow(0.3),
                            DecisionNode::leaf(DecisionAction::RequestHelp),
                            DecisionNode::leaf(DecisionAction::FullComputation),
                        ),
                    ),
                ),
            ),
        );

        Self::new(root)
    }

    /// Evaluate decision tree
    pub fn decide(&self, ctx: &DecisionContext) -> DecisionResult {
        self.traverse(&self.root, ctx, Vec::new())
    }

    fn traverse(
        &self,
        node: &DecisionNode,
        ctx: &DecisionContext,
        mut path: Vec<String>,
    ) -> DecisionResult {
        match node {
            DecisionNode::Leaf(action) => {
                path.push(format!("→ {:?}", action));
                DecisionResult {
                    action: action.clone(),
                    path,
                }
            }
            DecisionNode::Branch { condition, if_true, if_false } => {
                let result = condition.evaluate(ctx);
                path.push(format!("{:?} = {}", condition, result));
                if result {
                    self.traverse(if_true, ctx, path)
                } else {
                    self.traverse(if_false, ctx, path)
                }
            }
        }
    }
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::default_for_agent()
    }
}

/// Result of decision tree evaluation
#[derive(Debug, Clone)]
pub struct DecisionResult {
    pub action: DecisionAction,
    pub path: Vec<String>,
}

impl DecisionResult {
    pub fn path_string(&self) -> String {
        self.path.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_resources() {
        let tree = DecisionTree::default_for_agent();
        let ctx = DecisionContext {
            budget_remaining: 0.8,
            context_usage: 0.3,
            confidence: 0.9,
            ..Default::default()
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, DecisionAction::FullComputation);
    }

    #[test]
    fn test_low_budget() {
        let tree = DecisionTree::default_for_agent();
        let ctx = DecisionContext {
            budget_remaining: 0.05,
            ..Default::default()
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, DecisionAction::SimplifiedComputation);
    }

    #[test]
    fn test_high_failures() {
        let tree = DecisionTree::default_for_agent();
        let ctx = DecisionContext {
            failure_count: 5,
            budget_remaining: 0.8,
            ..Default::default()
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, DecisionAction::SafeFallback);
    }

    #[test]
    fn test_rate_limited() {
        let tree = DecisionTree::default_for_agent();
        let ctx = DecisionContext {
            is_rate_limited: true,
            budget_remaining: 0.8,
            ..Default::default()
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, DecisionAction::WaitRetry);
    }

    #[test]
    fn test_low_confidence() {
        let tree = DecisionTree::default_for_agent();
        let ctx = DecisionContext {
            budget_remaining: 0.8,
            confidence: 0.2,
            ..Default::default()
        };

        let result = tree.decide(&ctx);
        assert_eq!(result.action, DecisionAction::RequestHelp);
    }
}
