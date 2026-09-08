package memorylayer

// MemoryType is a cognitive memory type — how a memory is structured.
type MemoryType string

const (
	MemoryTypeEpisodic   MemoryType = "episodic"   // Specific events/interactions
	MemoryTypeSemantic   MemoryType = "semantic"   // Facts, concepts, relationships
	MemoryTypeProcedural MemoryType = "procedural" // How to do things
	MemoryTypeWorking    MemoryType = "working"    // Current task context
)

// MemorySubtype is a domain subtype — what a memory is about.
type MemorySubtype string

const (
	MemorySubtypeSolution    MemorySubtype = "solution"
	MemorySubtypeProblem     MemorySubtype = "problem"
	MemorySubtypeCodePattern MemorySubtype = "code_pattern"
	MemorySubtypeFix         MemorySubtype = "fix"
	MemorySubtypeError       MemorySubtype = "error"
	MemorySubtypeWorkflow    MemorySubtype = "workflow"
	MemorySubtypePreference  MemorySubtype = "preference"
	MemorySubtypeDecision    MemorySubtype = "decision"
	MemorySubtypeDirective   MemorySubtype = "directive"
	MemorySubtypeProfile     MemorySubtype = "profile"
	MemorySubtypeEntity      MemorySubtype = "entity"
	MemorySubtypeEvent       MemorySubtype = "event"
	MemorySubtypeInference   MemorySubtype = "inference"
)

// RecallMode is the retrieval strategy for recall queries.
type RecallMode string

const (
	RecallModeRAG    RecallMode = "rag"    // Fast vector similarity search
	RecallModeLLM    RecallMode = "llm"    // Deep semantic LLM-powered retrieval
	RecallModeHybrid RecallMode = "hybrid" // Combine both strategies
)

// SearchTolerance is the search precision level.
type SearchTolerance string

const (
	SearchToleranceLoose    SearchTolerance = "loose"
	SearchToleranceModerate SearchTolerance = "moderate"
	SearchToleranceStrict   SearchTolerance = "strict"
)

// RelationshipCategory is a high-level relationship category from the server's
// unified ontology.
type RelationshipCategory string

const (
	RelationshipCategoryHierarchical RelationshipCategory = "hierarchical"
	RelationshipCategoryCausal       RelationshipCategory = "causal"
	RelationshipCategoryTemporal     RelationshipCategory = "temporal"
	RelationshipCategorySimilarity   RelationshipCategory = "similarity"
	RelationshipCategoryLearning     RelationshipCategory = "learning"
	RelationshipCategoryReference    RelationshipCategory = "reference"
	RelationshipCategorySolution     RelationshipCategory = "solution"
	RelationshipCategoryContext      RelationshipCategory = "context"
	RelationshipCategoryWorkflow     RelationshipCategory = "workflow"
	RelationshipCategoryQuality      RelationshipCategory = "quality"
)

// RelationshipType is a specific relationship type between memories (snake_case,
// matching the server's ontology strings).
type RelationshipType string

const (
	// Causal
	RelationshipCauses   RelationshipType = "causes"
	RelationshipTriggers RelationshipType = "triggers"
	RelationshipLeadsTo  RelationshipType = "leads_to"
	RelationshipPrevents RelationshipType = "prevents"

	// Solution
	RelationshipSolves        RelationshipType = "solves"
	RelationshipAddresses     RelationshipType = "addresses"
	RelationshipAlternativeTo RelationshipType = "alternative_to"
	RelationshipImproves      RelationshipType = "improves"

	// Context
	RelationshipOccursIn  RelationshipType = "occurs_in"
	RelationshipAppliesTo RelationshipType = "applies_to"
	RelationshipWorksWith RelationshipType = "works_with"
	RelationshipRequires  RelationshipType = "requires"

	// Learning
	RelationshipBuildsOn    RelationshipType = "builds_on"
	RelationshipContradicts RelationshipType = "contradicts"
	RelationshipConfirms    RelationshipType = "confirms"
	RelationshipSupersedes  RelationshipType = "supersedes"

	// Similarity
	RelationshipSimilarTo RelationshipType = "similar_to"
	RelationshipVariantOf RelationshipType = "variant_of"
	RelationshipRelatedTo RelationshipType = "related_to"

	// Workflow
	RelationshipFollows   RelationshipType = "follows"
	RelationshipDependsOn RelationshipType = "depends_on"
	RelationshipEnables   RelationshipType = "enables"
	RelationshipBlocks    RelationshipType = "blocks"

	// Quality
	RelationshipEffectiveFor  RelationshipType = "effective_for"
	RelationshipPreferredOver RelationshipType = "preferred_over"
	RelationshipDeprecatedBy  RelationshipType = "deprecated_by"
)

// Stable constants shared with the MemoryLayer server. These mirror
// memorylayer_server.models.chat — the values MUST match the server.
const (
	// UserChatHomeWorkspace is the storage workspace sentinel for user-owned
	// chat threads. When a thread is created or appended-to with
	// ownership="user" (the SDK default), the caller-supplied workspace is
	// substituted with this sentinel so the thread lives in the cross-workspace
	// user namespace; the original workspace is preserved on per-message
	// metadata under MessageMetaAppWorkspaceKey.
	UserChatHomeWorkspace = "_user_chat"

	// MessageMetaAppWorkspaceKey is the reserved top-level key on chat message
	// metadata carrying the originating app workspace at write time.
	MessageMetaAppWorkspaceKey = "app_workspace"
)
