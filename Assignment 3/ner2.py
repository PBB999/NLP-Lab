import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Make sure you have these once:
# nltk.download("punkt")
# nltk.download("maxent_ne_chunker")
# nltk.download("words")
# nltk.download("averaged_perceptron_tagger")

piano_class_text = (
    "The future of India will be shaped by today’s younger generation who need quality education through digital literacy, making them productive and self-reliant citizens. "
    "'Digital literacy' is the skills required to achieve digital competence and use of Information and Communication Technology (ICT) for work, leisure, learning and communication. "
    "It does not replace traditional forms of literacy, instead complements and amplifies the skills that form the foundation of traditional forms. "
    "Learners can benefit from the knowledgebase and experience of 4 decades of Infosys as an enterprise. "
    "We also bring the quality content from our partners and leading universities across the world. "
    "Content is aligned with New Education policy 2020. "
    "In Mayfair or the City of London and has world-class piano instructors. "
    "Includes soft skills and vocational skills. "
    "Infosys Springboard is a Digital literacy program launched as part of the Infosys ESG Tech for Good charter. "
    "It aims to enable students and associated communities from early education to lifelong learners by imparting digital life skills through curated content & interventions, free of cost."
)

# Step 1: Tokenize & POS tagging
tokens = word_tokenize(piano_class_text)
pos_tags = pos_tag(tokens)

# Step 2: Named Entity Recognition
chunked = ne_chunk(pos_tags, binary=False)

# Extract named entities as text
entities = []
for subtree in chunked:
    if isinstance(subtree, Tree):
        entity_name = " ".join([token for token, pos in subtree.leaves()])
        entity_type = subtree.label()
        entities.append((entity_name, entity_type))

# Print extracted entities
for ent_name, ent_type in entities:
    print(f"Entity: {ent_name}\tType: {ent_type}")

# Optional: Visualize parse tree
chunked.draw()
