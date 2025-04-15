# Contributing Guide

Second Me is an open and friendly community. We are dedicated to building a collaborative, inspiring, and exuberant open source community for our members. Everyone is more than welcome to join our community to get help and to contribute to Second Me.

The Second Me community welcomes various forms of contributions, including code, non-code contributions, documentation, and more.

## How to Contribute

| Contribution Type | Details |
|------------------|---------|
| Report a bug | You can file an issue to report a bug with Second Me |
| Contribute code | You can contribute your code by fixing a bug or implementing a feature |
| Code Review | If you are an active contributor or committer of Second Me, you can help us review pull requests |
| Documentation | You can contribute documentation changes by fixing a documentation bug or proposing new content |

## Before Contributing
* Sign [CLA of Mindverse](https://cla-assistant.io/mindverse/Second-Me)
  
## Here is a checklist to prepare and submit your PR (pull request).
* Create your own GitHub branch by forking Second Me
* Checkout [README]() for how to deploy Second Me.
* Push changes to your personal fork.
* Create a PR with a detailed description, if commit messages do not express themselves.
* Submit PR for review and address all feedbacks.
* Wait for merging (done by committers).

## Development Workflow for External Contributors

As an external contributor, you'll need to follow the fork-based workflow. This ensures a safe and organized way to contribute to the project.

### 1. Fork the Repository
1. Visit https://github.com/Mindverse/Second-Me
2. Click the "Fork" button in the top-right corner
3. Select your GitHub account as the destination

### 2. Clone Your Fork
After forking, clone your fork to your local machine:
```bash
cd working_dir
# Replace USERNAME with your GitHub username
git clone git@github.com:USERNAME/Second-Me.git
cd Second-Me
```

### 3. Configure Upstream Remote
To keep your fork up to date with the main repository:
```bash
# Add the upstream repository
git remote add upstream git@github.com:Mindverse/Second-Me.git

# Verify your remotes
git remote -v
# You should see:
# origin    git@github.com:USERNAME/Second-Me.git (fetch)
# origin    git@github.com:USERNAME/Second-Me.git (push)
# upstream  git@github.com:Mindverse/Second-Me.git (fetch)
# upstream  git@github.com:Mindverse/Second-Me.git (push)
```

### 4. Keep Your Fork Updated
Before creating a new feature branch, ensure your fork's main branch is up to date:
```bash
# Switch to main branch
git checkout main

# Fetch upstream changes
git fetch upstream

# Rebase your main branch on top of upstream main
git rebase upstream/main

# Optional: Update your fork on GitHub
git push origin main
```

### 5. Create a Feature Branch
Always create a new branch for your changes:
```bash
# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

### 6. Make Your Changes
Make your desired changes in the feature branch. Make sure to:
- Follow the project's coding style
- Add tests if applicable
- Update documentation as needed

### 7. Commit Your Changes
```bash
# Add your changes
git add <filename>
# Or git add -A for all changes

# Commit with a clear message
git commit -m "feat: add new feature X"
```

### 8. Update Your Feature Branch
Before submitting your PR, update your feature branch with the latest changes:
```bash
# Fetch upstream changes
git fetch upstream

# Rebase your feature branch
git checkout feature/your-feature-name
git rebase upstream/main
```

### 9. Push to Your Fork
```bash
# Push your feature branch to your fork
git push origin feature/your-feature-name
```

### 10. Create a Pull Request
1. Visit your fork at `https://github.com/USERNAME/Second-Me`
2. Click "Compare & Pull Request"
3. Select:
   - Base repository: `Mindverse/Second-Me`
   - Base branch: `main`
   - Head repository: `USERNAME/Second-Me`
   - Compare branch: `feature/your-feature-name`
4. Fill in the PR template with:
   - Clear description of your changes
   - Any related issues
   - Testing steps if applicable

### 11. Review Process
1. Maintainers will review your PR
2. Address any feedback by:
   - Making requested changes
   - Pushing new commits to your feature branch
   - The PR will automatically update
3. Once approved, maintainers will merge your PR

### 12. CI Checks
- Automated checks will run on your PR
- All checks must pass before merging
- If checks fail, click "Details" to see why
- Fix any issues and push updates to your branch

## Tips for Successful Contributions
- Create focused, single-purpose PRs
- Follow the project's code style and conventions
- Write clear commit messages
- Keep your fork updated to avoid merge conflicts
- Be responsive during the review process
- Ask questions if anything is unclear
