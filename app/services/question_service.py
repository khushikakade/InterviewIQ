import random
from app.services.ai_service import ai_service

QUESTION_BANK = {
    'technical': {
        'python': [
            {
                "question_text": "Explain the difference between mutable and immutable data types in Python, and provide practical examples.",
                "type": "Technical",
                "category": "Python",
                "expected_concepts": ["mutable", "immutable", "list", "tuple", "memory reference", "id", "dictionary"]
            },
            {
                "question_text": "How do Python decorators work under the hood? Write or describe a simple timer decorator.",
                "type": "Technical",
                "category": "Python",
                "expected_concepts": ["first-class functions", "wrapper", "@syntax", "*args", "**kwargs", "functools.wraps"]
            },
            {
                "question_text": "Describe Python memory management, reference counting, and how the cyclic garbage collector operates.",
                "type": "Technical",
                "category": "Python",
                "expected_concepts": ["reference counting", "cyclic garbage collector", "gc module", "memory heap", "gil"]
            },
            {
                "question_text": "What are Python generators and yield statements, and how do they optimize memory usage?",
                "type": "Technical",
                "category": "Python",
                "expected_concepts": ["generator", "yield", "iterator", "lazy evaluation", "memory efficiency", "next()"]
            },
            {
                "question_text": "Explain the difference between *args and **kwargs in Python function definitions.",
                "type": "Technical",
                "category": "Python",
                "expected_concepts": ["positional arguments", "keyword arguments", "tuple unpacking", "dict unpacking", "variable parameters"]
            }
        ],
        'sql': [
            {
                "question_text": "Explain the difference between INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN with practical examples.",
                "type": "Technical",
                "category": "SQL",
                "expected_concepts": ["inner join", "left join", "null values", "matching rows", "unmatched rows", "outer join"]
            },
            {
                "question_text": "What are SQL window functions, and how do RANK(), DENSE_RANK(), and ROW_NUMBER() differ?",
                "type": "Technical",
                "category": "SQL",
                "expected_concepts": ["over() clause", "partition by", "order by", "rank vs dense_rank", "row_number"]
            },
            {
                "question_text": "How would you diagnose and optimize a slow-performing SQL query in a production database?",
                "type": "Technical",
                "category": "SQL",
                "expected_concepts": ["indexes", "explain plan", "avoid select *", "subquery vs join", "database normalization"]
            },
            {
                "question_text": "What is database ACID compliance, and why is atomicity crucial for transactional safety?",
                "type": "Technical",
                "category": "SQL",
                "expected_concepts": ["atomicity", "consistency", "isolation", "durability", "transactions", "commit", "rollback"]
            }
        ],
        'machine learning': [
            {
                "question_text": "Explain the bias-variance tradeoff in Machine Learning and how cross-validation and regularization prevent overfitting.",
                "type": "Technical",
                "category": "Machine Learning",
                "expected_concepts": ["underfitting", "overfitting", "regularization", "cross-validation", "model complexity", "l1/l2"]
            },
            {
                "question_text": "Walk me through how the Random Forest algorithm works and why ensemble bagging reduces variance.",
                "type": "Technical",
                "category": "Machine Learning",
                "expected_concepts": ["decision trees", "bagging", "bootstrap sampling", "random feature subset", "ensemble"]
            },
            {
                "question_text": "What metrics would you evaluate for an imbalanced classification model instead of plain accuracy?",
                "type": "Technical",
                "category": "Machine Learning",
                "expected_concepts": ["precision", "recall", "f1-score", "roc-auc", "confusion matrix", "smote", "pr-curve"]
            },
            {
                "question_text": "Explain the architecture of Convolutional Neural Networks (CNNs) and how pooling layers achieve spatial invariance.",
                "type": "Technical",
                "category": "Deep Learning",
                "expected_concepts": ["convolution", "kernels", "stride", "max pooling", "feature maps", "receptive field"]
            },
            {
                "question_text": "How do attention mechanisms and Transformers differ from traditional Recurrent Neural Networks (RNNs)?",
                "type": "Technical",
                "category": "NLP / AI",
                "expected_concepts": ["self-attention", "query key value", "parallel processing", "positional encoding", "transformers"]
            }
        ],
        'web': [
            {
                "question_text": "How do RESTful APIs differ from GraphQL, and in what architectural scenarios would you select one over the other?",
                "type": "Technical",
                "category": "Web Architecture",
                "expected_concepts": ["endpoints", "http methods", "overfetching", "schema", "statelessness", "graphql query"]
            },
            {
                "question_text": "Explain how React Virtual DOM works and how reconciliation optimizes browser rendering performance.",
                "type": "Technical",
                "category": "Frontend",
                "expected_concepts": ["virtual dom", "diffing algorithm", "reconciliation", "state updates", "component lifecycle"]
            },
            {
                "question_text": "What is event looping in Node.js, and how are non-blocking asynchronous I/O operations handled?",
                "type": "Technical",
                "category": "Backend",
                "expected_concepts": ["event loop", "call stack", "callback queue", "libuv", "promises", "async await"]
            }
        ],
        'devops': [
            {
                "question_text": "What is the difference between a virtual machine and a Docker container, and how do container layers work?",
                "type": "Technical",
                "category": "DevOps",
                "expected_concepts": ["dockerfile", "container image", "kernel sharing", "isolation", "hypervisor vs container"]
            },
            {
                "question_text": "Describe how a modern CI/CD pipeline automates testing, building, and zero-downtime deployment.",
                "type": "Technical",
                "category": "DevOps",
                "expected_concepts": ["github actions", "jenkins", "automated testing", "artifact registry", "blue-green deployment"]
            }
        ],
        'general': [
            {
                "question_text": "Walk me through the system architecture of a major software or machine learning project you built recently.",
                "type": "Technical",
                "category": "System Design",
                "expected_concepts": ["backend", "frontend", "database", "api design", "data flow", "scalability", "tradeoffs"]
            },
            {
                "question_text": "How do you select optimal data structures (e.g. HashMaps vs Trees) when optimizing time and space complexity?",
                "type": "Technical",
                "category": "Data Structures",
                "expected_concepts": ["hash table", "binary search tree", "time complexity", "o(1)", "o(log n)", "space complexity"]
            }
        ]
    },
    'behavioral': [
        {
            "question_text": "Describe a high-pressure situation where a technical project deadline was at risk. What specific steps did you take?",
            "type": "Behavioral",
            "category": "Time Management",
            "expected_concepts": ["situation", "task", "action", "result", "prioritization", "stakeholder communication"]
        },
        {
            "question_text": "Tell me about a technical disagreement or conflict you had with a teammate and how you reached a resolution.",
            "type": "Behavioral",
            "category": "Conflict Resolution",
            "expected_concepts": ["situation", "task", "action", "result", "empathy", "data-driven consensus", "collaboration"]
        },
        {
            "question_text": "Give an example of a technical mistake or system bug you caused in production. How did you handle accountability?",
            "type": "Behavioral",
            "category": "Accountability",
            "expected_concepts": ["accountability", "root cause analysis", "incident response", "prevention", "learning outcome"]
        },
        {
            "question_text": "Describe a project where you had to quickly learn an unfamiliar technology or framework to deliver results.",
            "type": "Behavioral",
            "category": "Adaptability",
            "expected_concepts": ["fast learning", "documentation", "prototyping", "action taken", "successful delivery"]
        }
    ],
    'hr': [
        {
            "question_text": "Tell me about your technical background, key accomplishments, and why you are interested in this role.",
            "type": "HR",
            "category": "Introduction",
            "expected_concepts": ["background", "core projects", "key skills", "career goals", "company alignment"]
        },
        {
            "question_text": "What are your top 2 technical strengths, and what is one technical area you are actively taking steps to improve?",
            "type": "HR",
            "category": "Self Awareness",
            "expected_concepts": ["core strengths", "growth mindset", "continuous learning", "self reflection"]
        },
        {
            "question_text": "Where do you envision your technical career progressing over the next 3 to 5 years?",
            "type": "HR",
            "category": "Career Vision",
            "expected_concepts": ["leadership", "technical mastery", "impact", "long-term vision"]
        }
    ]
}

class QuestionService:
    def generate_questions(self, role="Software Developer", interview_type="Mixed", difficulty="Medium", skills=None, count=5):
        """
        Generates personalized interview questions based on candidate profile.
        Uses AI API if configured, otherwise uses local deterministic question generation engine.
        """
        skills = [s.lower() for s in (skills or [])]

        prompt = f"""
        Generate {count} unique, non-repeating interview questions for a candidate applying for the role of '{role}'.
        Interview Type: {interview_type}
        Difficulty: {difficulty}
        Candidate Skills: {', '.join(skills) if skills else 'Software Engineering, Python, SQL'}

        Return a JSON array of objects, where each object has:
        - "question_text": The question string
        - "type": Question type (Technical, HR, or Behavioral)
        - "category": Category domain
        - "expected_concepts": List of key concepts/keywords expected in a good answer
        """

        fallback_questions = self._generate_local_questions(role, interview_type, difficulty, skills, count)

        # Query AI Service (with fallback)
        res = ai_service.generate_json(prompt, fallback_questions)

        if isinstance(res, list) and len(res) >= 1:
            questions = []
            for i, q in enumerate(res[:count]):
                q_text = q.get("question_text") or q.get("text")
                if not q_text or q_text.strip() == "":
                    q_text = fallback_questions[i]["question_text"] if i < len(fallback_questions) else f"Describe your approach to problem solving in {role}."

                questions.append({
                    "order_num": i + 1,
                    "question_text": q_text,
                    "question_type": q.get("type", interview_type if interview_type != "Mixed" else "Technical"),
                    "category": q.get("category", role),
                    "expected_concepts": q.get("expected_concepts", ["experience", "technical skills", "impact"])
                })
            return questions

        return fallback_questions

    def _generate_local_questions(self, role, interview_type, difficulty, skills, count):
        pool = []

        # 1. Tech questions based on candidate skills
        if interview_type in ['Technical', 'Mixed']:
            matched_tech = False
            for skill in skills:
                for key in QUESTION_BANK['technical']:
                    if key in skill and QUESTION_BANK['technical'][key]:
                        pool.extend(QUESTION_BANK['technical'][key])
                        matched_tech = True
            
            if not matched_tech:
                # Add general tech & python/sql questions as baseline
                pool.extend(QUESTION_BANK['technical']['python'])
                pool.extend(QUESTION_BANK['technical']['sql'])
                pool.extend(QUESTION_BANK['technical']['general'])

        # 2. Behavioral questions
        if interview_type in ['Behavioral', 'Mixed']:
            pool.extend(QUESTION_BANK['behavioral'])

        # 3. HR questions
        if interview_type in ['HR', 'Mixed']:
            pool.extend(QUESTION_BANK['hr'])

        # Remove duplicate question texts
        unique_pool = []
        seen_texts = set()
        for q in pool:
            if q["question_text"] not in seen_texts:
                seen_texts.add(q["question_text"])
                unique_pool.append(q)

        # Shuffle and select top `count`
        random.shuffle(unique_pool)
        selected = unique_pool[:count] if len(unique_pool) >= count else unique_pool

        # Format output
        questions = []
        for i, q in enumerate(selected):
            questions.append({
                "order_num": i + 1,
                "question_text": q["question_text"],
                "question_type": q["type"],
                "category": q["category"],
                "expected_concepts": q["expected_concepts"]
            })
        return questions

question_service = QuestionService()
