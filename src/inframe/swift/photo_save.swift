// inframe-photo-save: adds video files to the Photos library (library only: no album), with the add-only permission.
//
// PHAccessLevel.addOnly lets the app add photos and videos and never read the library, so no library-wide Photos permission
// and no Full Disk Access is involved. Built on first use by `inframe export --to-photos` (swiftc, no Xcode project) with
// photo_save_Info.plist embedded in the binary: macOS refuses to ask for Photos access without NSPhotoLibraryAddUsageDescription.
// Own code; links only Apple system frameworks.
//
//   inframe-photo-save FILE [FILE ...]     request add-only access (the system dialog shows the first time), add each file; JSON on stdout
//   inframe-photo-save --status            print the add-only authorization state as JSON; never asks, never adds
//   inframe-photo-save --version
//
// stdout: one {"row": {"path": ..., "saved": bool, "error": ...}} line per file, written the moment that file is done (so a caller that
//         times out or loses the helper still knows which videos were already added), then the summary:
//         {"authorization": "authorized"|"denied"|"restricted"|"notDetermined"|"unknown",
//          "status": "saved"|"partial"|"denied"|"failed",
//          "files": [{"path": ..., "saved": bool, "error": ...}], "message": ...}
// exit 0 when status is saved or partial, 3 when access was denied or restricted, 1 on any other failure, 2 for bad arguments.
import Foundation
import Photos

let version = "2"

func err(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

func authName(_ s: PHAuthorizationStatus) -> String {
    switch s {
    case .authorized, .limited: return "authorized"
    case .denied: return "denied"
    case .restricted: return "restricted"
    case .notDetermined: return "notDetermined"
    @unknown default: return "unknown"
    }
}

func emit(authorization: String, status: String, files: [[String: Any]], message: String) -> Never {
    let doc: [String: Any] = ["authorization": authorization, "status": status, "files": files, "message": message]
    let data = (try? JSONSerialization.data(withJSONObject: doc, options: [.sortedKeys])) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
    switch status {
    case "saved", "partial": exit(0)
    case "denied": exit(3)
    default: exit(1)
    }
}

func emitRow(_ row: [String: Any]) {
    let data = (try? JSONSerialization.data(withJSONObject: ["row": row], options: [.sortedKeys])) ?? Data("{}".utf8)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}

var paths: [String] = []
var statusOnly = false
for a in CommandLine.arguments.dropFirst() {
    switch a {
    case "--version": print("inframe-photo-save \(version)"); exit(0)
    case "--status": statusOnly = true
    default:
        if a.hasPrefix("--") { err("inframe-photo-save: unknown argument \(a)"); exit(2) }
        paths.append(a)
    }
}

if statusOnly {
    let s = PHPhotoLibrary.authorizationStatus(for: .addOnly)
    print("{\"authorization\": \"\(authName(s))\"}")
    exit(0)
}
if paths.isEmpty { err("inframe-photo-save: no files given"); exit(2) }

// 1. authorization (add-only). The handler runs on an arbitrary queue, so waiting on a semaphore here cannot deadlock.
var status = PHPhotoLibrary.authorizationStatus(for: .addOnly)
if status == .notDetermined {
    err("inframe-photo-save: asking macOS for permission to add to Photos (a dialog may be showing) ...")
    let sem = DispatchSemaphore(value: 0)
    PHPhotoLibrary.requestAuthorization(for: .addOnly) { s in status = s; sem.signal() }
    sem.wait()
}
let auth = authName(status)
guard status == .authorized || status == .limited else {
    let why = status == .restricted
        ? "Photos access is restricted on this Mac (parental controls or a device policy), so nothing was saved."
        : "Photos access was denied, so nothing was saved. Allow it in System Settings > Privacy & Security > Photos (add-only is enough), then run again."
    emit(authorization: auth, status: "denied",
         files: paths.map { ["path": $0, "saved": false, "error": "no permission to add to Photos"] }, message: why)
}

// 2. one change per file, so one bad file does not stop the others
var rows: [[String: Any]] = []
var nSaved = 0
for p in paths {
    let url = URL(fileURLWithPath: p)
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: p, isDirectory: &isDir), !isDir.boolValue else {
        let row: [String: Any] = ["path": p, "saved": false, "error": "no such file"]
        rows.append(row); emitRow(row); continue
    }
    do {
        try PHPhotoLibrary.shared().performChangesAndWait {
            let req = PHAssetCreationRequest.forAsset()
            let opts = PHAssetResourceCreationOptions()
            opts.shouldMoveFile = false  // copy: the exported file stays where it is
            opts.originalFilename = url.lastPathComponent
            req.addResource(with: .video, fileURL: url, options: opts)
        }
        nSaved += 1
        let row: [String: Any] = ["path": p, "saved": true]
        rows.append(row); emitRow(row)
    } catch {
        let row: [String: Any] = ["path": p, "saved": false, "error": error.localizedDescription]
        rows.append(row); emitRow(row)
    }
}
let st = nSaved == paths.count ? "saved" : (nSaved > 0 ? "partial" : "failed")
emit(authorization: auth, status: st, files: rows,
     message: st == "saved" ? "saved \(nSaved) to the Photos library"
         : "saved \(nSaved) of \(paths.count) to the Photos library")
