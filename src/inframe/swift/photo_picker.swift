// inframe-photo-picker: opens Apple's photo picker (PHPicker) on the shared Photos library, lets the person pick
// videos, exports the picked originals (current representation) into a folder and prints the paths as JSON.
//
// No Full Disk Access and no library-wide permission: PHPicker runs out of process and hands over only the picked items.
// Built on first use by `inframe photos pick` (swiftc, no Xcode project). Own code; links only Apple system frameworks.
//
//   inframe-photo-picker [--out DIR] [--limit N] [--images]     picker (videos, or photos with --images); JSON on stdout, progress and the start line on stderr
//   inframe-photo-picker --version
//
// stdout: {"cancelled": bool, "files": [{"path": ..., "name": ...}], "errors": [...]}   exit 0 (also when cancelled), 1 on failure.
import AppKit
import Foundation
import PhotosUI
import UniformTypeIdentifiers

let version = "1"

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func emit(cancelled: Bool, files: [[String: String]], errors: [String]) -> Never {
    let doc: [String: Any] = ["cancelled": cancelled, "files": files, "errors": errors]
    let data = (try? JSONSerialization.data(withJSONObject: doc, options: [.sortedKeys])) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    exit(errors.isEmpty || !files.isEmpty || cancelled ? 0 : 1)
}

var outDirArg: URL? = nil
var limit = 0  // 0 = no limit
var imagesOnly = false  // --images: pick photos (target photos) instead of videos
var selfTestCopy: URL? = nil  // --selftest-copy FILE: run the export copy step once for FILE as pick 1 (tests), no window
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--version": print("inframe-photo-picker \(version)"); exit(0)
    case "--out": outDirArg = args.isEmpty ? nil : URL(fileURLWithPath: args.removeFirst())
    case "--selftest-copy": selfTestCopy = args.isEmpty ? nil : URL(fileURLWithPath: args.removeFirst())
    case "--images": imagesOnly = true
    case "--limit": limit = Int(args.isEmpty ? "0" : args.removeFirst()) ?? 0
    default: err("inframe-photo-picker: unknown argument \(a)"); exit(2)
    }
}
guard let outDirChecked = outDirArg else { err("inframe-photo-picker: --out DIR is required"); exit(2) }
let outDir: URL = outDirChecked
do { try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) } catch {
    err("inframe-photo-picker: cannot create \(outDir.path): \(error.localizedDescription)"); exit(1)
}

func exportDir() -> URL { outDir }

/// Copies `url` to <out>/<NN>/<name>. The numbered folder is emptied first, so an --export-dir reused from an earlier run
/// neither makes copyItem fail on the existing file nor keeps stale files of other picks.
func exportCopy(of url: URL, number: Int) throws -> URL {
    let dir = exportDir().appendingPathComponent(String(format: "%02d", number), isDirectory: true)
    let fm = FileManager.default
    if fm.fileExists(atPath: dir.path) { try fm.removeItem(at: dir) }
    try fm.createDirectory(at: dir, withIntermediateDirectories: true)
    let dest = dir.appendingPathComponent(url.lastPathComponent)
    try fm.copyItem(at: url, to: dest)
    return dest
}

if let src = selfTestCopy {
    do { print(try exportCopy(of: src, number: 1).path); exit(0) } catch { err("inframe-photo-picker: \(error.localizedDescription)"); exit(1) }
}

final class Controller: NSObject, PHPickerViewControllerDelegate, NSWindowDelegate, NSApplicationDelegate {
    var window: NSWindow!
    let lock = NSLock()
    var files: [[String: String]] = []
    var errors: [String] = []
    var done = false

    func applicationDidFinishLaunching(_ n: Notification) {
        var cfg = PHPickerConfiguration(photoLibrary: PHPhotoLibrary.shared())
        cfg.filter = imagesOnly ? .images : .videos
        cfg.selectionLimit = limit          // 0 = unlimited (multiple selection)
        cfg.preferredAssetRepresentationMode = .current   // no transcoding: the file as stored (e.g. 120 / 240 fps)
        let picker = PHPickerViewController(configuration: cfg)
        picker.delegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 640),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        window.title = imagesOnly ? "inframe: pick photos" : "inframe: pick videos"
        window.contentViewController = picker
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowWillClose(_ n: Notification) {
        if !done { emit(cancelled: true, files: [], errors: []) }
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        if results.isEmpty { done = true; emit(cancelled: true, files: [], errors: []) }
        done = true
        window.orderOut(nil)
        err("inframe-photo-picker: exporting \(results.count) \(imagesOnly ? "photo(s)" : "video(s)")")
        let group = DispatchGroup()
        for (i, r) in results.enumerated() {
            let p = r.itemProvider
            let itemType = imagesOnly ? UTType.image : UTType.movie
            guard p.hasItemConformingToTypeIdentifier(itemType.identifier) else {
                note(error: "item \(i + 1) is not a \(imagesOnly ? "photo" : "video")"); continue
            }
            group.enter()
            _ = p.loadFileRepresentation(forTypeIdentifier: itemType.identifier) { url, error in
                defer { group.leave() }
                guard let url = url else { self.note(error: "item \(i + 1): \(error?.localizedDescription ?? "no file")"); return }
                // the file is deleted when this handler returns, so copy it now; a numbered folder keeps equal names apart
                do {
                    let dest = try exportCopy(of: url, number: i + 1)
                    self.lock.lock(); self.files.append(["path": dest.path, "name": url.lastPathComponent, "_i": String(i)]); self.lock.unlock()
                } catch { self.note(error: "item \(i + 1): \(error.localizedDescription)") }
            }
        }
        group.notify(queue: .main) {
            let ordered = self.files.sorted { Int($0["_i"]!)! < Int($1["_i"]!)! }.map { ["path": $0["path"]!, "name": $0["name"]!] }
            emit(cancelled: false, files: ordered, errors: self.errors)
        }
    }

    func note(error: String) { lock.lock(); errors.append(error); lock.unlock() }
}

err("inframe-photo-picker \(version): opening the Photos picker (shared library, \(imagesOnly ? "photos" : "videos") only)")
let app = NSApplication.shared
let controller = Controller()
app.delegate = controller
app.setActivationPolicy(.regular)
app.run()
