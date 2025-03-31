![Second Me](https://github.com/mindverse/Second-Me/blob/master/images/cover.png)

<div align="center">
  
[![Homepage](https://img.shields.io/badge/Second_Me-Homepage-blue?style=flat-square&logo=homebridge)](https://www.secondme.io/)
[![Report](https://img.shields.io/badge/Paper-arXiv-red?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2503.08102)
[![Discord](https://img.shields.io/badge/Chat-Discord-5865F2?style=flat-square&logo=discord&logoColor=white)](https://discord.gg/GpWHQNUwrg)
[![Twitter](https://img.shields.io/badge/Follow-@SecondMe_AI-1DA1F2?style=flat-square&logo=x&logoColor=white)](https://x.com/SecondMe_AI1)
[![Reddit](https://img.shields.io/badge/Join-Reddit-FF4500?style=flat-square&logo=reddit&logoColor=white)](https://www.reddit.com/r/SecondMeAI/)

</div>


## Our Vision

Companies like OpenAI built "Super AI" that threatens human independence. We crave individuality: AI that amplifies, not erases, you.

We’re challenging that with "**Second Me**": an open-source prototype where you craft your own **AI self**—a new AI species that preserves you, delivers your context, and defends your interests.

It’s **locally trained and hosted**—your data, your control—yet **globally connected**, scaling your intelligence across an AI network. Beyond that, it’s your AI identity interface—a bold standard linking your AI to the world, sparks collaboration among AI selves, and builds tomorrow’s truly native AI apps.

Join us. Tech enthusiasts, AI pros, domain experts—Second Me is your launchpad to extend your mind into the digital horizon.

## Key Features

### **Train Your AI Self** with AI-Native Memory ([Paper](https://arxiv.org/abs/2503.08102))
Start training your Second Me today with your own memories! Using Hierarchical Memory Modeling (HMM) and the Me-Alignment Algorithm, your AI self captures your identity, understands your context, and reflects you authentically.

 <p align="center">
  <img src="https://github.com/user-attachments/assets/a84c6135-26dc-4413-82aa-f4a373c0ff89" width="94%" />
</p>


### **Scale Your Intelligence** on the Second Me Network
Launch your AI self from your laptop onto our decentralized network—anyone or any app can connect with your permission, sharing your context as your digital identity.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a74a3f4-d8fd-41c1-8f24-534ed94c842a" width="94%" />
</p>


### Build Tomorrow’s Apps with Second Me
**Roleplay**: Your AI self switches personas to represent you in different scenarios.  
**AI Space**: Collaborate with other Second Mes to spark ideas or solve problems.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bc6125c1-c84f-4ecc-b620-8932cc408094" width="94%" />
</p>

### 100% **Privacy and Control**
Unlike traditional centralized AI systems, Second Me ensures that your information and intelligence remain local and completely private.



## Getting started & staying tuned with us
Star and join us, and you will receive all release notifications from GitHub without any delay!


 <p align="center">
  <img src="https://github.com/user-attachments/assets/5c14d956-f931-4c25-b0b3-3c2c96cd7581" width="94%" />
</p>

## Quick Start




### 🍎 Option 1: Local Setup (macOS with Apple Silicon)

<!-- > **Special Note: This section is specifically for Mac (Apple Silicon) users** -->

<details open>
<summary><b>Click to expand/collapse Mac setup details</b></summary>

##### Prerequisites
- Python 3.12 or higher
- Xcode Command Line Tools

##### Installing Xcode Command Line Tools
If you haven't installed Xcode Command Line Tools yet, you can install them by running:
```bash
xcode-select --install
```

After installation, you may need to accept the license agreement:
```bash
sudo xcodebuild -license accept
```

##### Setup Steps

1. **Clone the repository**
   ```bash
   git clone git@github.com:Mindverse/Second-Me.git
   cd Second-Me
   ```

2. **Set up the environment**

   Choose one of the following options:

   <details>
   <summary><b>Option A: For users with existing conda environment</b></summary>
   
   If you already have conda installed:
   
   1. Create a new environment from our environment file:
      ```bash
      conda env create -f environment.yml   # This will create an environment named 'second-me'
      conda activate second-me
      ```
   
   2. Set the custom conda mode in `.env`:
      ```bash
      CUSTOM_CONDA_MODE=true
      ```
   
   3. Run setup:
      ```bash
      make setup
      ```
   </details>

   <details>
   <summary><b>Option B: For new users</b></summary>
   
   If you're new or want a fresh environment:
   ```bash
   make setup
   ```
   
   This command will automatically:
   - Install all required system dependencies (including conda if not present)
   - Create a new Python environment named 'second-me'
   - Build llama.cpp
   - Set up frontend environment
   </details>

3. **Start the service**
   ```bash
   make start
   ```

</details>

### 🐳 Option 2: Docker Setup (For Linux & Windows users)


<details>
<summary><b>Click to expand/collapse Docker setup details</b></summary>

Docker provides a consistent environment across different operating systems.

##### Prerequisites
- Docker and Docker Compose installed on your system

> **Important:** You must install both Docker and Docker Compose before proceeding. If you haven't installed them yet:
> - For Docker installation: [Get Docker](https://docs.docker.com/get-docker/)
> - For Docker Compose installation: [Install Docker Compose](https://docs.docker.com/compose/install/)

##### Setup Steps

1. Clone the repository
```bash
git clone git@github.com:Mindverse/Second-Me.git
cd Second-Me
```

2. Build the Docker images
```bash
make docker-build
```

3. Start the containers
```bash
make docker-up
```

4. To stop the containers when you're done
```bash
make docker-down
```

##### Other Useful Docker Commands

- Restart all services
```bash
make docker-restart-all
```

- Rebuild and restart only the backend
```bash
make docker-restart-backend
```

- Rebuild and restart only the frontend
```bash
make docker-restart-frontend
```

</details>

### Accessing the Service

After starting the service (either with local setup or Docker), open your browser and visit:
```
http://localhost:3000
```

### View help and more commands
```bash
make help
```

### Important Notes
1. Ensure you have sufficient disk space (at least 10GB recommended)
2. If using local setup with an existing conda environment, ensure there are no conflicting package versions
3. First startup may take a few minutes to download and install dependencies
4. Some commands may require sudo privileges

### Troubleshooting
If you encounter issues, check:
1. For local setup: Python and Node.js versions meet requirements
2. For local setup: You're in the correct conda environment
3. All dependencies are properly installed
4. System firewall allows the application to use required ports
5. For Docker setup: Docker daemon is running and you have sufficient permissions

## Tutorial and Use Cases
🛠️ Feel free to follow [User tutorial](https://second-me.gitbook.io/a-new-ai-species-making-we-matter-again) to build your Second Me.

💡 Check out the links below to see how Second Me can be used in real-life scenarios:
- [Felix AMA (Roleplay app)](https://app.secondme.io/example/ama)
- [Brainstorming a 15-Day European City Itinerary (Network app)](https://app.secondme.io/example/brainstorming)
- [Icebreaking as a Speed Dating Match (Network app)](https://app.secondme.io/example/Icebreaker)

## Join the Community
- [Discord](https://discord.com/invite/GpWHQNUwrg)
- [Reddit](https://www.reddit.com/r/SecondMeAI/)
- [X](https://x.com/SecondMe_AI1)

## Coming Soon 

The following features have been completed internally and are being gradually integrated into the open-source project. For detailed experimental results and technical specifications, please refer to our [Technical Report](https://arxiv.org/abs/2503.08102).

### Model Enhancement Features
- [ ] **Long Chain-of-Thought Training Pipeline**: Enhanced reasoning capabilities through extended thought process training
- [ ] **Direct Preference Optimization for L2 Model**: Improved alignment with user preferences and intent
- [ ] **Data Filtering for Training**: Advanced techniques for higher quality training data selection
- [ ] **Apple Silicon Support**: Native support for Apple Silicon processors with MLX Training and Serving capabilities

### Product Features
- [ ] **Natural Language Memory Summarization**: Intuitive memory organization in natural language format


## Contributing

We welcome contributions to Second Me! Whether you're interested in fixing bugs, adding new features, or improving documentation, please check out our Contribution Guide. You can also support Second Me by sharing your experience with it in your community, at tech conferences, or on social media.

For more detailed information about development, please refer to our [Contributing Guide](./CONTRIBUTING.md).

## Contributors

We would like to express our gratitude to all the individuals who have contributed to Second Me! If you're interested in contributing to the future of intelligence uploading, whether through code, documentation, or ideas, please feel free to submit a pull request to our repository: [Second-Me](https://github.com/Mindverse/Second-Me).


<a href="https://github.com/mindverse/Second-Me/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=mindverse/Second-Me" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Acknowledgements

This work leverages the power of the open-source community. 

For data synthesis, we utilized [GraphRAG](https://github.com/microsoft/graphrag) from Microsoft.

For model deployment, we utilized [llama.cpp](https://github.com/ggml-org/llama.cpp), which provides efficient inference capabilities.

Our base models primarily come from the [Qwen2.5](https://huggingface.co/Qwen) series.

We also want to extend our sincere gratitude to all users who have experienced Second Me. We recognize that there is significant room for optimization throughout the entire pipeline, and we are fully committed to iterative improvements to ensure everyone can enjoy the best possible experience locally.

## License

Second Me is open source software licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

[license]: ./LICENSE

## Star History

<a href="https://www.star-history.com/#mindverse/Second-Me&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mindverse/Second-Me&type=Date" />
 </picture>
</a>
