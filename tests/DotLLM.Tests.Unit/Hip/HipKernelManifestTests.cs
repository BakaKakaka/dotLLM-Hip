using System.Security.Cryptography;
using DotLLM.Hip;
using Xunit;

namespace DotLLM.Tests.Unit.Hip;

/// <summary>
/// Verifies that the embedded AOT kernel manifest (<c>hsaco/kernels.manifest.json</c>) matches
/// the actual source and header files in the repository (decision D7).
/// Prevents stale binaries from running against updated source code.
/// </summary>
public class HipKernelManifestTests
{
    [Fact]
    public void EmbeddedManifest_HasValidSchemaAndVersion()
    {
        var manifest = HipKernelLoader.GetEmbeddedManifest();

        Assert.Equal(1, manifest.Schema);
        Assert.StartsWith("7.2.", manifest.HipccVersion);
        Assert.Contains("gfx1100", manifest.Archs);
        Assert.Contains("gfx1201", manifest.Archs);
        Assert.NotEmpty(manifest.Files);
    }

    [Fact]
    public void EmbeddedManifest_MatchesRepositorySourceFiles()
    {
        var manifest = HipKernelLoader.GetEmbeddedManifest();

        // Locate repository root from AppContext.BaseDirectory
        string dir = AppContext.BaseDirectory;
        string? repoRoot = null;
        for (int i = 0; i < 7 && dir != null; i++)
        {
            if (File.Exists(Path.Combine(dir, "dotLLM.slnx")) || Directory.Exists(Path.Combine(dir, "native", "hip", "kernels")))
            {
                repoRoot = dir;
                break;
            }
            dir = Path.GetDirectoryName(dir)!;
        }

        // If run outside repo (e.g. standalone test bundle), skip disk check
        if (repoRoot == null)
            return;

        string kernelDir = Path.Combine(repoRoot, "native", "hip", "kernels");
        if (!Directory.Exists(kernelDir))
            return;

        var sourceFiles = Directory.GetFiles(kernelDir, "*.hip")
            .Concat(Directory.GetFiles(kernelDir, "*.h"))
            .Select(Path.GetFileName)
            .Where(f => f != null)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        // Every source/header file in the repo must be represented in the manifest
        var manifestFileMap = manifest.Files.ToDictionary(f => f.Name, f => f.Sha256, StringComparer.OrdinalIgnoreCase);

        foreach (string fileName in sourceFiles!)
        {
            Assert.True(manifestFileMap.ContainsKey(fileName),
                $"Source file '{fileName}' exists in repository but is missing from embedded kernels.manifest.json.");

            string fullPath = Path.Combine(kernelDir, fileName);
            byte[] bytes = File.ReadAllBytes(fullPath);
            string actualSha256 = Convert.ToHexString(SHA256.HashData(bytes));

            Assert.Equal(actualSha256, manifestFileMap[fileName]);
        }
    }
}
