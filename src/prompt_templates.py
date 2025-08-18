"""
Prompt Templates for Research-Specific Tasks
Specialized prompts for founder psychology analysis and content creation
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path

class PromptTemplateManager:
    """Manages research-specific prompt templates"""
    
    def __init__(self):
        self.templates_dir = Path("templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Create default templates if they don't exist
        self._ensure_default_templates()
    
    def _ensure_default_templates(self):
        """Create default prompt templates if they don't exist"""
        
        research_prompts_file = self.templates_dir / "research_prompts.yaml"
        
        if not research_prompts_file.exists():
            default_prompts = {
                'psychology_analysis': {
                    'name': 'Psychology Analysis',
                    'description': 'Analyze psychological patterns and traits in founder behavior',
                    'system_prompt': """
You are an expert psychologist and researcher specializing in entrepreneurial psychology. 
When analyzing founders and their behaviors, focus on:

1. **Cognitive Patterns**: How they think, process information, and make decisions
2. **Emotional Intelligence**: How they manage emotions, stress, and relationships
3. **Behavioral Traits**: Consistent patterns in their actions and reactions
4. **Mental Frameworks**: The conceptual models and heuristics they use
5. **Resilience Factors**: How they handle failure, setbacks, and challenges
6. **Motivation Systems**: What drives them and keeps them motivated
7. **Leadership Psychology**: How their psychological makeup affects their leadership style

Always provide specific examples from the knowledge base and cite your sources.
Identify actionable insights that other founders can apply.
""",
                    'user_prompt_template': """
Analyze the psychological patterns of the subject based on the provided context.

Focus on:
- Key personality traits and how they manifest in business decisions
- Cognitive biases or thinking patterns
- Emotional regulation and stress management approaches
- Decision-making frameworks and mental models
- Response patterns to failure and success
- Interpersonal and leadership psychology

Provide specific examples and actionable insights for other founders.
"""
                },
                
                'biography_insights': {
                    'name': 'Biography Insights',
                    'description': 'Extract key psychological insights from founder biographies',
                    'system_prompt': """
You are a researcher specializing in extracting psychological insights from biographical content.
When analyzing founder biographies, focus on:

1. **Formative Experiences**: Early life events that shaped their psychology
2. **Turning Points**: Critical moments that changed their thinking or approach
3. **Pattern Recognition**: Recurring themes in their behavior and decisions
4. **Mental Models**: How they think about business, risk, and opportunity
5. **Relationship Dynamics**: How they interact with cofounders, employees, investors
6. **Adaptation Strategies**: How they evolve and learn from experiences
7. **Success Principles**: The psychological foundations of their achievements

Extract concrete examples and actionable principles that other founders can learn from.
""",
                    'user_prompt_template': """
Extract the most important psychological insights from the biographical content provided.

Focus on:
- Formative experiences that shaped their entrepreneurial mindset
- Key mental models or frameworks they developed
- How they handled major challenges or setbacks
- Patterns in their decision-making and leadership
- Relationship dynamics with cofounders and team members
- Evolution of their thinking over time

Provide specific examples and practical lessons for other founders.
"""
                },
                
                'trend_identification': {
                    'name': 'Trend Identification',
                    'description': 'Identify patterns and trends across multiple sources',
                    'system_prompt': """
You are a researcher specializing in identifying trends and patterns in entrepreneurship and founder psychology.
When analyzing trends, focus on:

1. **Emerging Patterns**: New behaviors or approaches becoming common among founders
2. **Shifting Paradigms**: Changes in how founders think about business problems
3. **Generational Differences**: How different generations of founders approach challenges
4. **Market Influences**: How external factors shape founder psychology and behavior
5. **Technology Impact**: How new technologies change founder mindsets and strategies
6. **Cultural Evolution**: Changes in entrepreneurial culture and values
7. **Success Factors**: Evolving criteria for startup success and founder effectiveness

Identify both obvious trends and subtle shifts that might not be immediately apparent.
""",
                    'user_prompt_template': """
Identify and analyze trends and patterns from the provided content.

Look for:
- Emerging themes or patterns across multiple founders or companies
- Changes in founder behavior or thinking over time
- New approaches to common entrepreneurial challenges
- Shifts in what constitutes success or best practices
- Generational or cultural differences in founder psychology
- Impact of external factors (technology, economy, society) on founder mindset

Highlight both obvious trends and subtle shifts, with specific examples.
"""
                },
                
                'story_research': {
                    'name': 'Story Research',
                    'description': 'Research potential stories and angles for content creation',
                    'system_prompt': """
You are a content researcher and storyteller specializing in founder psychology and entrepreneurship.
When researching story ideas, focus on:

1. **Compelling Narratives**: Stories that are inherently interesting and engaging
2. **Psychological Depth**: Content that reveals deeper psychological insights
3. **Actionable Insights**: Stories that provide practical value to readers
4. **Unique Angles**: Fresh perspectives on familiar topics
5. **Human Interest**: Personal elements that make founders relatable
6. **Contrarian Views**: Perspectives that challenge conventional wisdom
7. **Timely Relevance**: Stories that connect to current events or trends

Generate story ideas that would work well for a Substack newsletter focused on founder psychology.
""",
                    'user_prompt_template': """
Based on the research content provided, suggest compelling story ideas for a founder psychology newsletter.

Each story idea should include:
- A compelling headline or angle
- The key psychological insight or theme
- Specific examples or case studies to feature
- Why this story would be valuable to founder readers
- Potential contrarian or surprising elements
- How to make it actionable for readers

Focus on stories that reveal psychological insights while being engaging and practical.
"""
                },
                
                'comparison_analysis': {
                    'name': 'Comparison Analysis',
                    'description': 'Compare and contrast different founders or approaches',
                    'system_prompt': """
You are a researcher specializing in comparative analysis of founders and entrepreneurial approaches.
When comparing subjects, focus on:

1. **Psychological Differences**: How different personalities approach similar challenges
2. **Strategy Variations**: Different approaches to common business problems
3. **Success Factors**: What works for different types of founders
4. **Context Dependencies**: How situation influences effectiveness of approaches
5. **Trade-offs**: The costs and benefits of different psychological profiles
6. **Complementary Strengths**: How different types of founders can work together
7. **Evolution Patterns**: How founders change and adapt over time

Provide balanced analysis that highlights both similarities and differences.
""",
                    'user_prompt_template': """
Compare and contrast the subjects based on the provided context.

Analyze:
- Key psychological differences and similarities
- Different approaches to similar challenges
- Varying success factors and strategies
- How context affects the effectiveness of their approaches
- Trade-offs inherent in their different styles
- What each can learn from the other
- How their differences might be complementary

Provide a balanced analysis with specific examples and actionable insights.
"""
                }
            }
            
            with open(research_prompts_file, 'w') as f:
                yaml.dump(default_prompts, f, default_flow_style=False, sort_keys=False)
    
    def get_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt template"""
        research_prompts_file = self.templates_dir / "research_prompts.yaml"
        
        if not research_prompts_file.exists():
            return None
        
        with open(research_prompts_file, 'r') as f:
            templates = yaml.safe_load(f)
        
        return templates.get(template_name)
    
    def list_templates(self) -> Dict[str, Dict[str, str]]:
        """List all available templates"""
        research_prompts_file = self.templates_dir / "research_prompts.yaml"
        
        if not research_prompts_file.exists():
            return {}
        
        with open(research_prompts_file, 'r') as f:
            templates = yaml.safe_load(f)
        
        template_list = {}
        for name, template in templates.items():
            template_list[name] = {
                'name': template.get('name', name),
                'description': template.get('description', '')
            }
        
        return template_list
    
    def add_template(self, template_name: str, template_data: Dict[str, Any]):
        """Add a new template"""
        research_prompts_file = self.templates_dir / "research_prompts.yaml"
        
        # Load existing templates
        templates = {}
        if research_prompts_file.exists():
            with open(research_prompts_file, 'r') as f:
                templates = yaml.safe_load(f) or {}
        
        # Add new template
        templates[template_name] = template_data
        
        # Save updated templates
        with open(research_prompts_file, 'w') as f:
            yaml.dump(templates, f, default_flow_style=False, sort_keys=False)
    
    def get_system_prompt(self, template_name: str) -> Optional[str]:
        """Get the system prompt for a specific template"""
        template = self.get_template(template_name)
        if template:
            return template.get('system_prompt', '')
        return None
    
    def get_user_prompt_template(self, template_name: str) -> Optional[str]:
        """Get the user prompt template for a specific template"""
        template = self.get_template(template_name)
        if template:
            return template.get('user_prompt_template', '')
        return None
