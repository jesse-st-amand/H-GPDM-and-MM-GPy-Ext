class NodeStruct:
    """
    Node Structure for creating hierarchical models
    """
    def __init__(self):
        # create parent and child identifiers
        self.mChild = []
        self.mParent = None

    '''def __del__(self):
        # for all children, set the children's parent to null and set the child to null
        for child in self.mChild:
            if (child):
                child.SetParent(None)
                child = None'''

    def GetNumChildren(self):
        return len(self.mChild) - self.mChild.count(None)

    def AttachChild(self, child):
        if ~child:
            print('You cannot attach null children to a node')
            return -1
        # The child already has a parent
        if child.GetParent():
            print('The child already has a parent')
            return -1
        child.setParent(self)
        # insert the child in first available slot (if any)
        for i, current in enumerate(self.mChild):
            if current is None:
                self.mChild[i] = child
                return i
        # All slots are used, so append the child to the array
        numChildren = len(self.mChild)
        self.mChild.append(child)
        return numChildren

    def DetachChild(self, child):
        if child:
            for i,current in enumerate(self.mChild):  # Cycle through list of children
                if current == child:  # If current child in list is the target child, set current to null
                    current.SetParent(None)
                    current.pop(i)
                    return i

    def DetachChildAt(self, i):
        #check if child_index is in the list of children index bounds
        if (0 <= i & i < len(self.mChild)):
            child = self.mChild[i]
            if (child):
                child.SetParent(None)
                self.pop(i) #detach child at index
            return child
        return None

    def DetachAllChildren(self): #inefficient because of duplicate loop in DetachChild
        # cycle through list of children and detach
        for child in self.mChild:
            self.DetachChild(child)

    def SetChild(self, i, child):
        if child:
            if child.GetParent():
                print("The child already has a parent")
                return None
        numChildren = len(self.mChild)
        # check if index is in childList range
        if (0 <= i & i < numChildren):
            previousChild = self.mChild[i]
            if previousChild:
                previousChild.SetParent(None)
            if child:
                child.SetParent(self)
            self.mChild[i] = child
            return previousChild
        if child:
            child.SetParent(self)
        self.mChild.append(child)
        return None

    def GetChild(self, i):
        if (0 <= i & i < len(self.mChild)):
            return self.mChild[i]
        return None

    def GetParent(self):
        return self.mParent

    def SetParent(self, parent):
        self.mParent = parent







