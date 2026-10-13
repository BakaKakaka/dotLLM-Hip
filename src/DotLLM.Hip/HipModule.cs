using DotLLM.Hip.Interop;

namespace DotLLM.Hip;

/// <summary>
/// RAII wrapper around a loaded HIP module (<c>hipModule_t</c>).
/// Caches kernel function handles and safely unloads the module on disposal.
/// </summary>
public sealed class HipModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new(StringComparer.Ordinal);

    /// <summary>The raw native module handle.</summary>
    public nint Handle => _module;

    internal HipModule(nint module)
    {
        _module = module;
    }

    /// <summary>
    /// Looks up a kernel function by name in the module, caching the function pointer.
    /// </summary>
    /// <param name="name">Name of the kernel function (<c>extern "C"</c>).</param>
    /// <returns>Native function pointer handle.</returns>
    public nint GetFunction(string name)
    {
        ObjectDisposedException.ThrowIf(_module == 0, this);

        if (!_functions.TryGetValue(name, out nint func))
        {
            HipApi.hipModuleGetFunction(out func, _module, name).ThrowOnError();
            _functions[name] = func;
        }

        return func;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint module = Interlocked.Exchange(ref _module, 0);
        if (module != 0)
        {
            HipApi.hipModuleUnload(module);
            _functions.Clear();
        }
    }
}
