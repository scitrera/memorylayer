module github.com/scitrera/memorylayer/memorylayer-sdk-go/aether

go 1.25.14

require (
	github.com/scitrera/aether/api v0.2.3
	github.com/scitrera/aether/sdk/go v0.2.3
	github.com/scitrera/memorylayer/memorylayer-sdk-go v0.2.0
)

require (
	github.com/bradenaw/juniper v0.10.0 // indirect
	github.com/scitrera/go-backpressure v0.1.1 // indirect
	golang.org/x/exp v0.0.0-20220217172124-1812c5b45e43 // indirect
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/text v0.37.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260414002931-afd174a4e478 // indirect
	google.golang.org/grpc v1.82.1 // indirect
	google.golang.org/protobuf v1.36.11 // indirect
)

// Intra-repo: the parent SDK module lives one directory up. Consumers ignore
// this (replace applies only to the main module) and resolve the tagged
// version instead; it exists so the two modules build together from a checkout.
replace github.com/scitrera/memorylayer/memorylayer-sdk-go => ../
